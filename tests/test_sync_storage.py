import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from optibot_scraper.sync_storage import (
    LocalSyncStore,
    SpacesSyncStore,
    missing_spaces_env_vars,
    sync_store_from_env,
)


class MissingObjectError(Exception):
    response = {
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }


class FakeSpacesClient:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Bucket, Key):
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise MissingObjectError() from exc
        return {"Body": io.BytesIO(body)}

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body
        return {"ContentType": ContentType}


class SyncStorageTests(unittest.TestCase):
    def test_local_sync_store_reads_and_writes_state_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LocalSyncStore(
                state_path=root / "state" / "sync_state.json",
                runs_dir=root / "runs",
            )
            summary = {"run_started_at": "2026-01-01T00:00:00Z"}

            self.assertEqual(store.read_state({"articles": {}}), {"articles": {}})
            store.write_state({"vector_store_id": "vs_1", "articles": {}})
            store.write_run_artifacts(summary)

            self.assertEqual(store.read_state({})["vector_store_id"], "vs_1")
            self.assertTrue((root / "runs" / "latest.json").exists())
            self.assertEqual(len(list((root / "runs").glob("*.json"))), 2)

    def test_spaces_sync_store_reads_missing_state_as_default_and_writes_remote_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = FakeSpacesClient()
            store = SpacesSyncStore(
                local_store=LocalSyncStore(
                    state_path=root / "state" / "sync_state.json",
                    runs_dir=root / "runs",
                ),
                bucket="kb-state",
                prefix="daily/job",
                client=client,
            )

            self.assertEqual(store.read_state({"articles": {}}), {"articles": {}})
            store.write_state({"vector_store_id": "vs_1", "articles": {}})
            store.write_run_artifacts({"run_started_at": "2026-01-01T00:00:00Z"})

            self.assertEqual(store.read_state({})["vector_store_id"], "vs_1")
            self.assertIn(("kb-state", "daily/job/job_state/sync_state.json"), client.objects)
            self.assertIn(("kb-state", "daily/job/job_runs/latest.json"), client.objects)
            self.assertTrue((root / "state" / "sync_state.json").exists())

    def test_partial_spaces_env_is_rejected(self):
        with patch.dict("os.environ", {"SPACES_BUCKET": "bucket"}, clear=True):
            self.assertIn("SPACES_ENDPOINT_URL", missing_spaces_env_vars())
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, "partially configured"):
                    sync_store_from_env(root / "state.json", root / "runs")

    def test_full_spaces_env_builds_spaces_store(self):
        env = {
            "SPACES_ENDPOINT_URL": "https://sgp1.digitaloceanspaces.com",
            "SPACES_REGION": "sgp1",
            "SPACES_BUCKET": "bucket",
            "SPACES_ACCESS_KEY_ID": "key",
            "SPACES_SECRET_ACCESS_KEY": "secret",
            "SPACES_PREFIX": "prefix",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("optibot_scraper.sync_storage.build_spaces_client", return_value=FakeSpacesClient()):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    store = sync_store_from_env(root / "state.json", root / "runs")

        self.assertIsInstance(store, SpacesSyncStore)
        self.assertEqual(store.bucket, "bucket")
        self.assertEqual(store.prefix, "prefix")


if __name__ == "__main__":
    unittest.main()
