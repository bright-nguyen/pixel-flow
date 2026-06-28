from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from optibot_scraper.scraper import (
    DEFAULT_BASE_URL,
    DEFAULT_LOCALE,
    DEFAULT_OUTPUT_DIR,
    content_hash,
    fetch_articles,
    write_articles,
)
from optibot_scraper.vector_store import (
    DEFAULT_CHUNK_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PREPARED_CHUNK_TARGET,
    DEFAULT_VECTOR_STORE_NAME,
    chunking_strategy,
    load_env_file,
    prepare_chunks,
)
from optibot_scraper.sync_storage import LocalSyncStore, SpacesSyncStore, sync_store_from_env


DEFAULT_STATE_PATH = Path("data/job_state/sync_state.json")
DEFAULT_RUNS_DIR = Path("data/job_runs")
DEFAULT_UPLOAD_CONCURRENCY = 5


@dataclass(frozen=True)
class DeltaPlan:
    added: list[str]
    updated: list[str]
    deleted: list[str]
    skipped: list[str]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_current_articles_state(
    articles_dir: Path,
    chunk_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunk_manifest.get("chunks", []):
        chunks_by_source.setdefault(chunk["source_path"], []).append(
            {
                "path": chunk["path"],
                "tokens": chunk["tokens"],
                "hash": content_hash(Path(chunk["path"]).read_text(encoding="utf-8")),
            }
        )

    current: dict[str, dict[str, Any]] = {}
    for source_path_string, chunks in chunks_by_source.items():
        source_path = Path(source_path_string)
        text = source_path.read_text(encoding="utf-8")
        metadata = article_metadata_from_markdown(text)
        article_id = metadata["article_id"] or source_path.stem.split("-", 1)[0]
        current[article_id] = {
            "article_id": article_id,
            "title": metadata["title"],
            "url": metadata["article_url"],
            "updated_at": metadata["last_updated"],
            "source_path": source_path.as_posix(),
            "source_hash": content_hash(text),
            "chunks": chunks,
            "file_ids": [],
        }
    return current


def article_metadata_from_markdown(text: str) -> dict[str, str]:
    metadata = {
        "title": "",
        "article_url": "",
        "article_id": "",
        "last_updated": "",
    }
    for line in text.splitlines():
        if line == "---":
            break
        if line.startswith("# ") and not metadata["title"]:
            metadata["title"] = line[2:].strip()
        elif line.startswith("Article URL:"):
            metadata["article_url"] = line.split(":", 1)[1].strip()
        elif line.startswith("Article ID:"):
            metadata["article_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Last Updated:"):
            metadata["last_updated"] = line.split(":", 1)[1].strip()
    return metadata


def classify_delta(
    previous_articles: dict[str, dict[str, Any]],
    current_articles: dict[str, dict[str, Any]],
) -> DeltaPlan:
    previous_ids = set(previous_articles)
    current_ids = set(current_articles)

    added = sorted(current_ids - previous_ids)
    deleted = sorted(previous_ids - current_ids)
    updated = []
    skipped = []

    for article_id in sorted(previous_ids & current_ids):
        if (
            previous_articles[article_id].get("source_hash")
            != current_articles[article_id].get("source_hash")
        ):
            updated.append(article_id)
        else:
            skipped.append(article_id)

    return DeltaPlan(added=added, updated=updated, deleted=deleted, skipped=skipped)


def ensure_vector_store(
    client: OpenAI,
    vector_store_id: str | None,
    vector_store_name: str,
) -> str:
    if vector_store_id:
        return vector_store_id

    vector_store = client.vector_stores.create(
        name=vector_store_name,
        description="OptiSigns support articles synced by the daily job.",
        metadata={"project": "optibot_takehome", "sync": "daily"},
    )
    return vector_store.id


def optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if not value:
        return None
    return value


def delete_uploaded_file(client: OpenAI, vector_store_id: str, file_id: str) -> bool:
    deleted_cleanly = True
    try:
        client.vector_stores.files.delete(file_id, vector_store_id=vector_store_id)
    except Exception:
        deleted_cleanly = False

    try:
        client.files.delete(file_id)
    except Exception:
        deleted_cleanly = False

    return deleted_cleanly


def ensure_file_batch_completed(file_batch: Any, file_ids: list[str]) -> None:
    status = getattr(file_batch, "status", None)
    file_counts = getattr(file_batch, "file_counts", None)
    failed = file_count_value(file_counts, "failed")
    cancelled = file_count_value(file_counts, "cancelled")
    if status != "completed" or failed or cancelled:
        raise RuntimeError(
            "OpenAI vector-store batch indexing failed: "
            f"status={status}, completed={file_count_value(file_counts, 'completed')}, "
            f"failed={failed}, cancelled={cancelled}, total={file_count_value(file_counts, 'total')}, "
            f"file_ids={file_ids}"
        )


def file_count_value(file_counts: Any, field: str) -> int:
    if file_counts is None:
        return 0
    if isinstance(file_counts, dict):
        return int(file_counts.get(field) or 0)
    return int(getattr(file_counts, field, 0) or 0)


def upload_chunk_file(client: OpenAI, chunk_path: Path) -> str:
    with chunk_path.open("rb") as stream:
        file_obj = client.files.create(file=stream, purpose="assistants")
    return file_obj.id


def upload_article_chunks_batch(
    client: OpenAI,
    vector_store_id: str,
    article_states: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    max_concurrency: int,
) -> list[str]:
    upload_items: list[tuple[str, int, Path]] = []
    for article_state in article_states:
        article_id = article_state["article_id"]
        for chunk_index, chunk in enumerate(article_state.get("chunks", []), start=1):
            upload_items.append((article_id, chunk_index, Path(chunk["path"])))

    if not upload_items:
        return []

    strategy = chunking_strategy(chunk_size, chunk_overlap)
    print(
        f"Uploading {len(upload_items)} delta chunks before batch attach "
        f"(max_concurrency={max_concurrency}).",
        flush=True,
    )

    uploaded_by_position: list[str | None] = [None] * len(upload_items)
    try:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(upload_chunk_file, client, chunk_path): index
                for index, (_, _, chunk_path) in enumerate(upload_items)
            }
            for completed_count, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                article_id, chunk_index, _ = upload_items[index]
                file_id = future.result()
                uploaded_by_position[index] = file_id
                print(
                    f"[{completed_count}/{len(upload_items)}] Uploaded chunk "
                    f"{chunk_index} for article {article_id}: {file_id}",
                    flush=True,
                )

        file_ids = [file_id for file_id in uploaded_by_position if file_id is not None]
        print(f"Attaching {len(file_ids)} uploaded chunks as one OpenAI batch.", flush=True)
        file_batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vector_store_id,
            file_ids=file_ids,
            chunking_strategy=strategy,
        )
        ensure_file_batch_completed(file_batch, file_ids)
    except Exception:
        for file_id in [file_id for file_id in uploaded_by_position if file_id]:
            delete_uploaded_file(client, vector_store_id, file_id)
        raise

    position = 0
    for article_state in article_states:
        chunk_count = len(article_state.get("chunks", []))
        article_state["file_ids"] = file_ids[position : position + chunk_count]
        position += chunk_count

    return file_ids


def build_run_summary(
    started_at: str,
    finished_at: str,
    vector_store_id: str | None,
    delta: DeltaPlan,
    uploaded_chunks: int,
    deleted_chunks: int,
    failed: int = 0,
    dry_run: bool = False,
    cleanup_failed_file_ids: list[str] | None = None,
) -> dict[str, Any]:
    cleanup_failed_file_ids = cleanup_failed_file_ids or []
    return {
        "dry_run": dry_run,
        "run_started_at": started_at,
        "run_finished_at": finished_at,
        "vector_store_id": vector_store_id,
        "added": len(delta.added),
        "updated": len(delta.updated),
        "deleted": len(delta.deleted),
        "skipped": len(delta.skipped),
        "uploaded_chunks": uploaded_chunks,
        "deleted_chunks": deleted_chunks,
        "failed": failed,
        "cleanup_failed_file_count": len(cleanup_failed_file_ids),
        "cleanup_failed_file_ids": cleanup_failed_file_ids,
        "added_article_ids": delta.added,
        "updated_article_ids": delta.updated,
        "deleted_article_ids": delta.deleted,
    }


def write_run_artifacts(runs_dir: Path, summary: dict[str, Any]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json(runs_dir / "latest.json", summary)
    timestamp = summary["run_started_at"].replace(":", "").replace(".", "-")
    write_json(runs_dir / f"{timestamp}.json", summary)


def printable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if not key.endswith("_article_ids") and not key.endswith("_file_ids")
    }


def run_sync(
    *,
    base_url: str = DEFAULT_BASE_URL,
    locale: str = DEFAULT_LOCALE,
    articles_dir: Path = DEFAULT_OUTPUT_DIR,
    chunks_dir: Path = DEFAULT_CHUNK_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    vector_store_name: str = DEFAULT_VECTOR_STORE_NAME,
    vector_store_id: str | None = None,
    limit: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    prepared_chunk_target: int = DEFAULT_PREPARED_CHUNK_TARGET,
    max_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
    dry_run: bool = False,
    client: OpenAI | None = None,
    store: LocalSyncStore | SpacesSyncStore | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    if limit is not None and not dry_run:
        raise ValueError(
            "--limit is only allowed with --dry-run for sync jobs. "
            "A partial non-dry-run sync would treat omitted articles as deleted."
        )
    store = store or sync_store_from_env(state_path, runs_dir)

    limit_label = f", limit={limit}" if limit is not None else ""
    print(f"Fetching articles from {base_url} ({locale}{limit_label})...", flush=True)
    result = fetch_articles(base_url=base_url, locale=locale, limit=limit)
    articles = result.articles
    print(f"Fetched {len(articles)} articles. Writing Markdown...", flush=True)
    write_articles(articles, output_dir=articles_dir, clean=True)
    print("Preparing full-corpus chunks...", flush=True)
    chunk_manifest = prepare_chunks(
        source_dir=articles_dir,
        output_dir=chunks_dir,
        max_tokens=prepared_chunk_target,
        clean=True,
    )
    print(
        f"Prepared {chunk_manifest['chunk_file_count']} chunks "
        f"from {chunk_manifest['source_file_count']} source files.",
        flush=True,
    )

    previous_state = store.read_state(default={"articles": {}})
    previous_articles = previous_state.get("articles", {})
    current_articles = build_current_articles_state(articles_dir, chunk_manifest)
    delta = classify_delta(previous_articles, current_articles)

    existing_vector_store_id = (
        vector_store_id
        or previous_state.get("vector_store_id")
        or optional_env("OPENAI_VECTOR_STORE_ID")
    )

    uploaded_chunks = sum(
        len(current_articles[article_id]["chunks"])
        for article_id in delta.added + delta.updated
    )
    planned_deleted_chunks = sum(
        len(previous_articles[article_id].get("file_ids", []))
        for article_id in delta.updated + delta.deleted
    )
    print(
        "Delta: "
        f"added={len(delta.added)}, updated={len(delta.updated)}, "
        f"deleted={len(delta.deleted)}, skipped={len(delta.skipped)}.",
        flush=True,
    )

    if dry_run:
        summary = build_run_summary(
            started_at,
            utc_now(),
            existing_vector_store_id,
            delta,
            uploaded_chunks=uploaded_chunks,
            deleted_chunks=planned_deleted_chunks,
            dry_run=True,
        )
        store.write_run_artifacts(summary)
        print(json.dumps(printable_summary(summary), indent=2, ensure_ascii=False))
        return summary

    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for sync")
        client = OpenAI(api_key=api_key)

    synced_vector_store_id = ensure_vector_store(
        client,
        existing_vector_store_id,
        vector_store_name,
    )
    print(f"Using vector store {synced_vector_store_id}.", flush=True)

    next_articles = {
        article_id: article
        for article_id, article in previous_articles.items()
        if article_id not in delta.deleted
    }

    upload_article_ids = delta.added + delta.updated
    upload_article_states = [
        dict(current_articles[article_id])
        for article_id in upload_article_ids
    ]
    upload_article_chunks_batch(
        client,
        synced_vector_store_id,
        upload_article_states,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_concurrency=max_concurrency,
    )
    for article_state in upload_article_states:
        next_articles[article_state["article_id"]] = article_state

    for article_id in delta.skipped:
        next_articles[article_id] = previous_articles[article_id]

    next_state = {
        "vector_store_id": synced_vector_store_id,
        "last_successful_run_at": utc_now(),
        "article_count": len(next_articles),
        "articles": dict(sorted(next_articles.items())),
    }
    store.write_state(next_state)

    cleanup_failed_file_ids: list[str] = []
    actual_deleted_chunks = 0
    cleanup_file_ids = list(previous_state.get("cleanup_failed_file_ids", []))
    stale_article_ids = delta.updated + delta.deleted
    for index, article_id in enumerate(stale_article_ids, start=1):
        print(
            f"[{index}/{len(stale_article_ids)}] Deleting stale files for article {article_id}",
            flush=True,
        )
        cleanup_file_ids.extend(previous_articles[article_id].get("file_ids", []))

    for file_id in dict.fromkeys(cleanup_file_ids):
        deleted_cleanly = delete_uploaded_file(client, synced_vector_store_id, file_id)
        if deleted_cleanly:
            actual_deleted_chunks += 1
        else:
            cleanup_failed_file_ids.append(file_id)

    if cleanup_failed_file_ids:
        next_state["cleanup_failed_file_ids"] = cleanup_failed_file_ids
        store.write_state(next_state)

    summary = build_run_summary(
        started_at,
        next_state["last_successful_run_at"],
        synced_vector_store_id,
        delta,
        uploaded_chunks=uploaded_chunks,
        deleted_chunks=actual_deleted_chunks,
        failed=len(cleanup_failed_file_ids),
        cleanup_failed_file_ids=cleanup_failed_file_ids,
    )
    store.write_run_artifacts(summary)
    print(json.dumps(printable_summary(summary), indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OptiBot daily sync job.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--locale", default=DEFAULT_LOCALE)
    parser.add_argument("--articles-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--vector-store-name", default=DEFAULT_VECTOR_STORE_NAME)
    parser.add_argument("--vector-store-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--prepared-chunk-target", type=int, default=DEFAULT_PREPARED_CHUNK_TARGET)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_UPLOAD_CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_args()
    try:
        run_sync(
            base_url=args.base_url,
            locale=args.locale,
            articles_dir=args.articles_dir,
            chunks_dir=args.chunks_dir,
            state_path=args.state_path,
            runs_dir=args.runs_dir,
            vector_store_name=args.vector_store_name,
            vector_store_id=args.vector_store_id,
            limit=args.limit,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            prepared_chunk_target=args.prepared_chunk_target,
            max_concurrency=args.max_concurrency,
            dry_run=args.dry_run,
        )
    except (RuntimeError, ValueError) as exc:
        print(exc)
        return 1
    return 0
