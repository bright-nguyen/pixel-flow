from __future__ import annotations

import argparse
import json
import math
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken
from openai import OpenAI


DEFAULT_SOURCE_DIR = Path("data/articles")
DEFAULT_CHUNK_DIR = Path("data/chunks")
DEFAULT_INPUT_DIR = DEFAULT_CHUNK_DIR
DEFAULT_REPORT_PATH = Path("data/vector_store_upload_report.json")
DEFAULT_VECTOR_STORE_NAME = "OptiBot Support Articles"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_PREPARED_CHUNK_TARGET = 900
DEFAULT_PREPARED_CHUNK_OVERLAP_BLOCKS = 1

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
EMBEDDED_MEDIA_PATTERN = re.compile(r"\[(Embedded media:[^\]]+)\]\([^)]+\)")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class FileEstimate:
    path: Path
    bytes: int
    tokens: int
    estimated_chunks: int


@dataclass(frozen=True)
class PreparedChunk:
    source_path: Path
    section_path: str
    text: str


def load_env_file(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def markdown_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(path for path in input_dir.glob("*.md") if path.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def query_terms(query: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in STOP_WORDS and len(term) > 1
    ]
    return terms or re.findall(r"[a-z0-9]+", query.lower())


def score_file_for_query(
    path: Path,
    query: str,
    term_weights: dict[str, float] | None = None,
) -> float:
    text = path.read_text(encoding="utf-8").lower()
    metadata, _ = extract_article_metadata(text)
    title = metadata.get("title", "").lower()
    stem = path.stem.lower().replace("-", " ")
    terms = query_terms(query)
    score = 0.0
    term_weights = term_weights or {term: 1.0 for term in terms}

    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    searchable_title = " ".join(re.findall(r"[a-z0-9]+", f"{title} {stem}"))
    searchable_text = " ".join(re.findall(r"[a-z0-9]+", text))
    if normalized_query and normalized_query in searchable_text:
        score += 100

    for term in terms:
        weight = term_weights.get(term, 1.0)
        title_count = searchable_title.count(term)
        body_count = searchable_text.count(term)
        if title_count:
            score += 250 * weight * title_count
        score += min(body_count, 10) * weight
    return score


def query_term_weights(files: list[Path], query: str) -> dict[str, float]:
    terms = query_terms(query)
    if not terms:
        return {}

    documents = [
        " ".join(re.findall(r"[a-z0-9]+", path.read_text(encoding="utf-8").lower()))
        for path in files
    ]
    weights = {}
    for term in terms:
        document_frequency = sum(1 for document in documents if term in document)
        weights[term] = len(documents) / max(document_frequency, 1)
    return weights


def select_markdown_files(
    input_dir: Path,
    query: str | None = None,
    min_sources: int = 30,
    limit: int | None = None,
) -> list[Path]:
    files = markdown_files(input_dir)
    if query:
        weights = query_term_weights(files, query)
        scored = [
            (score_file_for_query(path, query, term_weights=weights), path)
            for path in files
        ]
        scored.sort(key=lambda item: (-item[0], item[1].name))
        selected_count = min(max(min_sources, 1), len(scored))
        files = [path for _, path in scored[:selected_count]]
    if limit is not None:
        files = files[:limit]
    return files


def validate_chunking(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size < 100 or chunk_size > 4096:
        raise ValueError("--chunk-size must be between 100 and 4096 tokens")
    if chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be greater than or equal to 0")
    if chunk_overlap > chunk_size // 2:
        raise ValueError("--chunk-overlap must not exceed half of --chunk-size")


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def extract_article_metadata(text: str) -> tuple[dict[str, str], str]:
    parts = text.split("\n---\n", 1)
    header = parts[0]
    body = parts[1] if len(parts) == 2 else text

    metadata = {
        "title": "",
        "article_url": "",
        "article_id": "",
        "last_updated": "",
    }
    for line in header.splitlines():
        if line.startswith("# ") and not metadata["title"]:
            metadata["title"] = line[2:].strip()
        elif line.startswith("Article URL:"):
            metadata["article_url"] = line.split(":", 1)[1].strip()
        elif line.startswith("Article ID:"):
            metadata["article_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Last Updated:"):
            metadata["last_updated"] = line.split(":", 1)[1].strip()

    return metadata, body.strip()


def image_placeholder(match: re.Match[str]) -> str:
    alt_text = match.group(1).strip()
    if alt_text.lower().startswith("image:"):
        return f"[{alt_text}]"
    if alt_text:
        return f"[Image: {alt_text}]"
    return "[Image]"


def strip_image_links(markdown: str) -> str:
    markdown = IMAGE_PATTERN.sub(image_placeholder, markdown)
    return EMBEDDED_MEDIA_PATTERN.sub(r"[\1]", markdown)


def section_path_from_stack(title: str, heading_stack: list[tuple[int, str]]) -> str:
    if not heading_stack:
        return title
    return " > ".join(text for _, text in heading_stack)


def split_markdown_sections(path: Path) -> list[tuple[dict[str, str], str, str]]:
    text = path.read_text(encoding="utf-8")
    metadata, body = extract_article_metadata(text)
    title = metadata["title"] or path.stem
    sections = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path = title
    in_code_fence = False

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((metadata, current_path, strip_image_links(content)))

    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence

        heading_match = HEADING_PATTERN.match(line) if not in_code_fence else None
        if heading_match:
            flush()
            current_lines = [line]
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = [
                item for item in heading_stack if item[0] < level
            ] + [(level, heading_text)]
            current_path = section_path_from_stack(title, heading_stack)
            continue

        current_lines.append(line)

    flush()
    return sections


def split_blocks(markdown: str) -> list[str]:
    blocks = []
    current = []
    in_code_fence = False

    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
        if not in_code_fence and not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def chunk_section_text(
    text: str,
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    if count_tokens(text, encoding_name=encoding_name) <= max_tokens:
        return [text]

    chunks = []
    blocks = split_blocks(text)
    current: list[str] = []

    for block in blocks:
        candidate = "\n\n".join(current + [block]).strip()
        if current and count_tokens(candidate, encoding_name=encoding_name) > max_tokens:
            chunks.append("\n\n".join(current).strip())
            current = current[-DEFAULT_PREPARED_CHUNK_OVERLAP_BLOCKS:] if current else []

        if count_tokens(block, encoding_name=encoding_name) > max_tokens:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
            chunks.extend(split_large_block(block, max_tokens, encoding_name))
        else:
            current.append(block)

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def split_large_block(
    block: str,
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[str]:
    chunks = []
    current: list[str] = []
    for line in block.splitlines():
        candidate = "\n".join(current + [line]).strip()
        if current and count_tokens(candidate, encoding_name=encoding_name) > max_tokens:
            chunks.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def render_prepared_chunk(
    metadata: dict[str, str],
    source_path: Path,
    section_path: str,
    content: str,
) -> str:
    lines = [
        f"# {metadata.get('title') or source_path.stem}",
        "",
    ]
    if metadata.get("article_url"):
        lines.extend([f"Article URL: {metadata['article_url']}", ""])
    if metadata.get("article_id"):
        lines.extend([f"Article ID: {metadata['article_id']}", ""])
    if metadata.get("last_updated"):
        lines.extend([f"Last Updated: {metadata['last_updated']}", ""])
    lines.extend(
        [
            f"Source File: {source_path.name}",
            "",
            f"Section Path: {section_path}",
            "",
            "---",
            "",
            content.strip(),
            "",
        ]
    )
    return "\n".join(lines)


def combined_section_path(section_paths: list[str]) -> str:
    unique_paths = []
    for section_path in section_paths:
        if section_path not in unique_paths:
            unique_paths.append(section_path)

    if len(unique_paths) == 1:
        return unique_paths[0]
    return "Multiple sections: " + " | ".join(unique_paths)


def group_section_chunks(
    pieces: list[tuple[dict[str, str], str, str]],
    max_tokens: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[tuple[dict[str, str], str, str]]:
    groups = []
    current_metadata: dict[str, str] | None = None
    current_paths: list[str] = []
    current_texts: list[str] = []

    def flush() -> None:
        nonlocal current_metadata, current_paths, current_texts
        if current_metadata and current_texts:
            groups.append(
                (
                    current_metadata,
                    combined_section_path(current_paths),
                    "\n\n".join(current_texts).strip(),
                )
            )
        current_metadata = None
        current_paths = []
        current_texts = []

    for metadata, section_path, text in pieces:
        candidate_texts = current_texts + [text]
        candidate = "\n\n".join(candidate_texts).strip()
        if current_texts and count_tokens(candidate, encoding_name=encoding_name) > max_tokens:
            flush()

        if current_metadata is None:
            current_metadata = metadata
        current_paths.append(section_path)
        current_texts.append(text)

    flush()
    return groups


def prepare_chunks(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_dir: Path = DEFAULT_CHUNK_DIR,
    max_tokens: int = DEFAULT_PREPARED_CHUNK_TARGET,
    clean: bool = True,
    limit: int | None = None,
    query: str | None = None,
    min_sources: int = 30,
    encoding_name: str = DEFAULT_ENCODING,
) -> dict[str, Any]:
    files = select_markdown_files(
        source_dir,
        query=query,
        min_sources=min_sources,
        limit=limit,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for path in output_dir.glob("*.md"):
            path.unlink()

    chunk_records = []
    for source_path in files:
        article_pieces: list[tuple[dict[str, str], str, str]] = []
        for metadata, section_path, section_text in split_markdown_sections(source_path):
            for chunk_text in chunk_section_text(
                section_text,
                max_tokens=max_tokens,
                encoding_name=encoding_name,
            ):
                article_pieces.append((metadata, section_path, chunk_text))

        article_chunks = [
            (
                section_path,
                render_prepared_chunk(metadata, source_path, section_path, chunk_text),
            )
            for metadata, section_path, chunk_text in group_section_chunks(
                article_pieces,
                max_tokens=max_tokens,
                encoding_name=encoding_name,
            )
        ]

        for index, (section_path, chunk_text) in enumerate(article_chunks, start=1):
            chunk_path = output_dir / f"{source_path.stem}--chunk-{index:03d}.md"
            chunk_path.write_text(chunk_text, encoding="utf-8")
            chunk_records.append(
                {
                    "path": chunk_path.as_posix(),
                    "source_path": source_path.as_posix(),
                    "section_path": section_path,
                    "tokens": count_tokens(chunk_text, encoding_name=encoding_name),
                }
            )

    manifest = {
        "source_dir": source_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "query": query,
        "min_sources": min_sources if query else None,
        "source_file_count": len(files),
        "source_files": [path.as_posix() for path in files],
        "chunk_file_count": len(chunk_records),
        "max_tokens": max_tokens,
        "chunks": chunk_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def estimate_chunk_count(token_count: int, chunk_size: int, chunk_overlap: int) -> int:
    if token_count <= 0:
        return 0
    if token_count <= chunk_size:
        return 1

    stride = chunk_size - chunk_overlap
    return 1 + math.ceil((token_count - chunk_size) / stride)


def estimate_files(
    files: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[FileEstimate]:
    estimates = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        tokens = count_tokens(text, encoding_name=encoding_name)
        estimates.append(
            FileEstimate(
                path=path,
                bytes=path.stat().st_size,
                tokens=tokens,
                estimated_chunks=estimate_chunk_count(
                    tokens,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
            )
        )
    return estimates


def chunking_strategy(chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
    return {
        "type": "static",
        "static": {
            "max_chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
        },
    }


def model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


def build_report(
    estimates: list[FileEstimate],
    chunk_size: int,
    chunk_overlap: int,
    vector_store_id: str | None = None,
    file_batch: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "vector_store_id": vector_store_id,
        "file_batch_id": getattr(file_batch, "id", None),
        "file_batch_status": getattr(file_batch, "status", None),
        "file_counts": model_dump(getattr(file_batch, "file_counts", None)),
        "local_file_count": len(estimates),
        "total_bytes": sum(item.bytes for item in estimates),
        "total_tokens": sum(item.tokens for item in estimates),
        "estimated_chunk_count": sum(item.estimated_chunks for item in estimates),
        "chunking_strategy": chunking_strategy(chunk_size, chunk_overlap),
        "files": [
            {
                "path": item.path.as_posix(),
                "bytes": item.bytes,
                "tokens": item.tokens,
                "estimated_chunks": item.estimated_chunks,
            }
            for item in estimates
        ],
    }


def file_count_value(file_counts: Any, field: str) -> int:
    if file_counts is None:
        return 0
    if isinstance(file_counts, dict):
        return int(file_counts.get(field) or 0)
    return int(getattr(file_counts, field, 0) or 0)


def file_batch_error(file_batch: Any) -> str | None:
    status = getattr(file_batch, "status", None)
    file_counts = getattr(file_batch, "file_counts", None)
    failed = file_count_value(file_counts, "failed")
    cancelled = file_count_value(file_counts, "cancelled")
    total = file_count_value(file_counts, "total")
    completed = file_count_value(file_counts, "completed")

    if status == "completed" and failed == 0 and cancelled == 0:
        return None

    return (
        "OpenAI vector-store file batch did not complete successfully: "
        f"status={status}, completed={completed}, failed={failed}, "
        f"cancelled={cancelled}, total={total}"
    )


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_prepare_chunks_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare section-aware Markdown chunks for OptiBot RAG upload."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_PREPARED_CHUNK_TARGET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--query",
        default=None,
        help="Select a relevant subset of source articles for this user question.",
    )
    parser.add_argument(
        "--min-sources",
        type=int,
        default=30,
        help="Minimum source articles to keep when --query is used.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing chunk files in the output directory.",
    )
    return parser.parse_args()


def prepare_chunks_main() -> int:
    args = parse_prepare_chunks_args()
    manifest = prepare_chunks(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        clean=not args.no_clean,
        limit=args.limit,
        query=args.query,
        min_sources=args.min_sources,
    )
    print(
        f"Prepared {manifest['chunk_file_count']} chunk files "
        f"from {manifest['source_file_count']} source files in {args.output_dir}."
    )
    return 0


def upload_files(
    client: OpenAI,
    files: list[Path],
    vector_store_name: str,
    existing_vector_store_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
    max_concurrency: int,
) -> tuple[str, Any]:
    strategy = chunking_strategy(chunk_size, chunk_overlap)
    if existing_vector_store_id:
        vector_store_id = existing_vector_store_id
        print(f"Using vector store {vector_store_id}.", flush=True)
    else:
        print(f"Creating vector store {vector_store_name!r}...", flush=True)
        vector_store = client.vector_stores.create(
            name=vector_store_name,
            description="OptiSigns support articles converted to Markdown.",
            metadata={"project": "optibot_takehome", "source": "support.optisigns.com"},
        )
        vector_store_id = vector_store.id
        print(f"Created vector store {vector_store_id}.", flush=True)

    print(
        f"Uploading {len(files)} files as an OpenAI batch "
        f"(max_concurrency={max_concurrency}). Waiting for indexing...",
        flush=True,
    )
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("rb")) for path in files]
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id,
            files=streams,
            max_concurrency=max_concurrency,
            chunking_strategy=strategy,
        )
    print(
        f"Batch finished with status={getattr(batch, 'status', None)}.",
        flush=True,
    )

    return vector_store_id, batch


def parse_upload_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload scraped OptiSigns Markdown files to an OpenAI Vector Store."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--vector-store-name", default=DEFAULT_VECTOR_STORE_NAME)
    parser.add_argument(
        "--vector-store-id",
        default=None,
        help="Reuse an existing vector store. Defaults to OPENAI_VECTOR_STORE_ID.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-files", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env_file()
    args = parse_upload_args()
    validate_chunking(args.chunk_size, args.chunk_overlap)

    files = markdown_files(args.input_dir, limit=args.limit)
    if len(files) < args.min_files:
        print(f"Expected at least {args.min_files} Markdown files, found {len(files)}.")
        return 1

    estimates = estimate_files(
        files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.dry_run:
        report = build_report(
            estimates,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            dry_run=True,
        )
        write_report(report, args.report_path)
        print(
            f"Dry run: {report['local_file_count']} files, "
            f"{report['estimated_chunk_count']} estimated chunks. "
            f"Wrote {args.report_path}."
        )
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required. Copy .env.sample to .env and set it.")
        return 1

    vector_store_id = args.vector_store_id or os.environ.get("OPENAI_VECTOR_STORE_ID")
    client = OpenAI(api_key=api_key)
    vector_store_id, file_batch = upload_files(
        client,
        files,
        vector_store_name=args.vector_store_name,
        existing_vector_store_id=vector_store_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_concurrency=args.max_concurrency,
    )

    report = build_report(
        estimates,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_id=vector_store_id,
        file_batch=file_batch,
    )
    completion_error = file_batch_error(file_batch)
    if completion_error:
        report["error"] = completion_error
        write_report(report, args.report_path)
        print(completion_error)
        print(f"Wrote {args.report_path}.")
        return 1

    write_report(report, args.report_path)

    print(
        f"Uploaded {report['local_file_count']} files to vector store "
        f"{vector_store_id}. File batch status: {report['file_batch_status']}. "
        f"Estimated chunks: {report['estimated_chunk_count']}. "
        f"Wrote {args.report_path}."
    )
    return 0
