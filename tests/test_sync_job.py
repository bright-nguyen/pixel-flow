import io
import json
import threading
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from optibot_scraper.scraper import Article, FetchResult
from optibot_scraper.sync_job import classify_delta, run_sync, write_json


class MemorySyncStore:
    def __init__(self):
        self.state = None
        self.run_artifacts = []

    def read_state(self, default):
        return json.loads(json.dumps(self.state if self.state is not None else default))

    def write_state(self, payload):
        self.state = json.loads(json.dumps(payload))

    def write_run_artifacts(self, summary):
        self.run_artifacts.append(json.loads(json.dumps(summary)))


class MockVectorStoreFiles:
    def __init__(
        self,
        events,
        attach_status="completed",
        attach_statuses=None,
        delete_fail_ids=None,
    ):
        self.events = events
        self.attach_status = attach_status
        self.attach_statuses = list(attach_statuses or [])
        self.delete_fail_ids = set(delete_fail_ids or [])
        self.attached = []
        self.deleted = []

    def create_and_poll(self, file_id, *, vector_store_id, chunking_strategy):
        self.events.append(("attach", file_id))
        self.attached.append((vector_store_id, file_id, chunking_strategy))
        status = self.attach_statuses.pop(0) if self.attach_statuses else self.attach_status
        return SimpleNamespace(id=file_id, status=status)

    def delete(self, file_id, *, vector_store_id):
        self.events.append(("vector_delete", file_id))
        if file_id in self.delete_fail_ids:
            raise RuntimeError(f"delete failed for {file_id}")
        self.deleted.append((vector_store_id, file_id))
        return SimpleNamespace(id=file_id, deleted=True)


class MockFileBatches:
    def __init__(self, events, batch_status="completed", batch_statuses=None):
        self.events = events
        self.batch_status = batch_status
        self.batch_statuses = list(batch_statuses or [])
        self.attached = []

    def create_and_poll(self, *, vector_store_id, file_ids, chunking_strategy):
        file_ids = list(file_ids)
        self.events.append(("batch_attach", tuple(file_ids)))
        self.attached.append((vector_store_id, file_ids, chunking_strategy))
        status = self.batch_statuses.pop(0) if self.batch_statuses else self.batch_status
        failed = 0 if status == "completed" else len(file_ids)
        completed = len(file_ids) if status == "completed" else 0
        return SimpleNamespace(
            id="batch_mock",
            status=status,
            file_counts=SimpleNamespace(
                total=len(file_ids),
                completed=completed,
                failed=failed,
                cancelled=0,
            ),
        )


class MockVectorStores:
    def __init__(
        self,
        events,
        attach_status="completed",
        attach_statuses=None,
        delete_fail_ids=None,
    ):
        self.files = MockVectorStoreFiles(
            events,
            attach_status=attach_status,
            attach_statuses=attach_statuses,
            delete_fail_ids=delete_fail_ids,
        )
        self.file_batches = MockFileBatches(
            events,
            batch_status=attach_status,
            batch_statuses=attach_statuses,
        )
        self.created = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="vs_mock")


class MockFiles:
    def __init__(self, events):
        self.events = events
        self.created = []
        self.deleted = []
        self.next_id = 1
        self.lock = threading.Lock()

    def create(self, *, file, purpose):
        with self.lock:
            file_id = f"file_{self.next_id}"
            self.next_id += 1
            self.events.append(("file_create", file_id))
            self.created.append((file.name, purpose, file_id))
        return SimpleNamespace(id=file_id)

    def delete(self, file_id):
        self.events.append(("file_delete", file_id))
        self.deleted.append(file_id)
        return SimpleNamespace(id=file_id, deleted=True)


class MockOpenAIClient:
    def __init__(
        self,
        attach_status="completed",
        attach_statuses=None,
        vector_delete_fail_ids=None,
    ):
        self.events = []
        self.vector_stores = MockVectorStores(
            self.events,
            attach_status=attach_status,
            attach_statuses=attach_statuses,
            delete_fail_ids=vector_delete_fail_ids,
        )
        self.files = MockFiles(self.events)


class SyncJobTests(unittest.TestCase):
    def test_classify_delta_identifies_added_updated_deleted_skipped(self):
        previous = {
            "1": {"source_hash": "same"},
            "2": {"source_hash": "old"},
            "3": {"source_hash": "deleted"},
        }
        current = {
            "1": {"source_hash": "same"},
            "2": {"source_hash": "new"},
            "4": {"source_hash": "added"},
        }

        delta = classify_delta(previous, current)

        self.assertEqual(delta.added, ["4"])
        self.assertEqual(delta.updated, ["2"])
        self.assertEqual(delta.deleted, ["3"])
        self.assertEqual(delta.skipped, ["1"])

    def test_run_sync_dry_run_writes_summary_without_uploading(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-01T00:00:00Z",
            body="<h2>Setup</h2><p>Hello</p>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    summary = run_sync(
                        articles_dir=root / "articles",
                        chunks_dir=root / "chunks",
                        state_path=root / "state" / "sync_state.json",
                        runs_dir=root / "runs",
                        dry_run=True,
                    )

            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["added"], 1)
            self.assertEqual(summary["uploaded_chunks"], 1)
            self.assertTrue((root / "runs" / "latest.json").exists())
            self.assertFalse((root / "state" / "sync_state.json").exists())

    def test_run_sync_updates_existing_article_and_deletes_old_file(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-02T00:00:00Z",
            body="<h2>Setup</h2><p>New body</p>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state" / "sync_state.json"
            write_json(
                state_path,
                {
                    "vector_store_id": "vs_existing",
                    "articles": {
                        "1": {
                            "article_id": "1",
                            "source_hash": "old_hash",
                            "file_ids": ["file_old"],
                        }
                    },
                },
            )
            client = MockOpenAIClient()

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    summary = run_sync(
                        articles_dir=root / "articles",
                        chunks_dir=root / "chunks",
                        state_path=state_path,
                        runs_dir=root / "runs",
                        client=client,
                    )

            self.assertEqual(summary["updated"], 1)
            self.assertEqual(client.vector_stores.files.deleted, [("vs_existing", "file_old")])
            self.assertEqual(client.files.deleted, ["file_old"])
            self.assertEqual(len(client.files.created), 1)
            self.assertEqual(len(client.vector_stores.file_batches.attached), 1)
            self.assertLess(
                client.events.index(("batch_attach", ("file_1",))),
                client.events.index(("vector_delete", "file_old")),
            )

    def test_run_sync_rejects_limit_for_non_dry_run(self):
        with patch("optibot_scraper.sync_job.fetch_articles") as fetch_articles:
            with self.assertRaisesRegex(ValueError, "--limit is only allowed"):
                run_sync(
                    limit=1,
                    dry_run=False,
                    client=MockOpenAIClient(),
                )

        fetch_articles.assert_not_called()

    def test_run_sync_keeps_old_files_when_replacement_indexing_fails(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-02T00:00:00Z",
            body="<h2>Setup</h2><p>New body</p>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state" / "sync_state.json"
            write_json(
                state_path,
                {
                    "vector_store_id": "vs_existing",
                    "articles": {
                        "1": {
                            "article_id": "1",
                            "source_hash": "old_hash",
                            "file_ids": ["file_old"],
                        }
                    },
                },
            )
            client = MockOpenAIClient(attach_status="failed")

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "batch indexing failed"):
                        run_sync(
                            articles_dir=root / "articles",
                            chunks_dir=root / "chunks",
                            state_path=state_path,
                            runs_dir=root / "runs",
                            client=client,
                        )

            self.assertEqual(client.files.deleted, ["file_1"])
            self.assertNotIn(("vector_delete", "file_old"), client.events)

    def test_run_sync_rolls_back_uploaded_chunks_when_batch_indexing_fails(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-02T00:00:00Z",
            body=(
                "<h2>Part 1</h2><p>" + ("Alpha " * 100) + "</p>"
                "<h2>Part 2</h2><p>" + ("Beta " * 100) + "</p>"
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state" / "sync_state.json"
            write_json(
                state_path,
                {
                    "vector_store_id": "vs_existing",
                    "articles": {
                        "1": {
                            "article_id": "1",
                            "source_hash": "old_hash",
                            "file_ids": ["file_old"],
                        }
                    },
                },
            )
            client = MockOpenAIClient(attach_status="failed")

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "batch indexing failed"):
                        run_sync(
                            articles_dir=root / "articles",
                            chunks_dir=root / "chunks",
                            state_path=state_path,
                            runs_dir=root / "runs",
                            prepared_chunk_target=20,
                            client=client,
                        )

            created_file_ids = {
                file_id
                for _, _, file_id in client.files.created
            }
            self.assertGreaterEqual(len(created_file_ids), 2)
            self.assertEqual(set(client.files.deleted), created_file_ids)
            batch_events = [
                event
                for event in client.events
                if event[0] == "batch_attach"
            ]
            self.assertEqual(len(batch_events), 1)
            self.assertEqual(set(batch_events[0][1]), created_file_ids)
            self.assertNotIn(("vector_delete", "file_old"), client.events)
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))["articles"]["1"][
                    "file_ids"
                ],
                ["file_old"],
            )

    def test_run_sync_persists_new_state_when_stale_cleanup_fails(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-02T00:00:00Z",
            body="<h2>Setup</h2><p>New body</p>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state" / "sync_state.json"
            write_json(
                state_path,
                {
                    "vector_store_id": "vs_existing",
                    "articles": {
                        "1": {
                            "article_id": "1",
                            "source_hash": "old_hash",
                            "file_ids": ["file_old"],
                        }
                    },
                },
            )
            client = MockOpenAIClient(vector_delete_fail_ids={"file_old"})

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    summary = run_sync(
                        articles_dir=root / "articles",
                        chunks_dir=root / "chunks",
                        state_path=state_path,
                        runs_dir=root / "runs",
                        client=client,
                    )

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["cleanup_failed_file_ids"], ["file_old"])
            self.assertEqual(summary["deleted_chunks"], 0)
            self.assertEqual(state["articles"]["1"]["file_ids"], ["file_1"])
            self.assertEqual(state["cleanup_failed_file_ids"], ["file_old"])

    def test_run_sync_retries_previous_cleanup_failures(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-01T00:00:00Z",
            body="<h2>Setup</h2><p>Hello</p>",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "state" / "sync_state.json"

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    first_summary = run_sync(
                        articles_dir=root / "articles",
                        chunks_dir=root / "chunks",
                        state_path=state_path,
                        runs_dir=root / "runs",
                        client=MockOpenAIClient(),
                    )

            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["cleanup_failed_file_ids"] = ["file_stale"]
            write_json(state_path, state)
            retry_client = MockOpenAIClient()

            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    retry_summary = run_sync(
                        articles_dir=root / "articles",
                        chunks_dir=root / "chunks",
                        state_path=state_path,
                        runs_dir=root / "runs",
                        client=retry_client,
                    )

            retry_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(first_summary["added"], 1)
            self.assertEqual(retry_summary["skipped"], 1)
            self.assertEqual(retry_summary["deleted_chunks"], 1)
            self.assertEqual(retry_summary["cleanup_failed_file_count"], 0)
            self.assertNotIn("cleanup_failed_file_ids", retry_state)
            self.assertIn(("vector_delete", "file_stale"), retry_client.events)
            self.assertIn(("file_delete", "file_stale"), retry_client.events)

    def test_run_sync_uses_persisted_store_state_across_runs(self):
        article = Article(
            article_id=1,
            title="Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Article",
            updated_at="2026-01-01T00:00:00Z",
            body="<h2>Setup</h2><p>Hello</p>",
        )
        store = MemorySyncStore()

        with tempfile.TemporaryDirectory() as first_tmp:
            first_root = Path(first_tmp)
            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    first_summary = run_sync(
                        articles_dir=first_root / "articles",
                        chunks_dir=first_root / "chunks",
                        state_path=first_root / "state" / "sync_state.json",
                        runs_dir=first_root / "runs",
                        client=MockOpenAIClient(),
                        store=store,
                    )

        with tempfile.TemporaryDirectory() as second_tmp:
            second_root = Path(second_tmp)
            second_client = MockOpenAIClient()
            with patch(
                "optibot_scraper.sync_job.fetch_articles",
                return_value=FetchResult([article], skipped_count=0),
            ):
                with redirect_stdout(io.StringIO()):
                    second_summary = run_sync(
                        articles_dir=second_root / "articles",
                        chunks_dir=second_root / "chunks",
                        state_path=second_root / "state" / "sync_state.json",
                        runs_dir=second_root / "runs",
                        client=second_client,
                        store=store,
                    )

        self.assertEqual(first_summary["added"], 1)
        self.assertEqual(second_summary["skipped"], 1)
        self.assertEqual(second_summary["uploaded_chunks"], 0)
        self.assertEqual(second_client.files.created, [])
        self.assertEqual(len(store.run_artifacts), 2)


if __name__ == "__main__":
    unittest.main()
