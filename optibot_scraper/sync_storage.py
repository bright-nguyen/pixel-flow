from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_SPACES_ENV_VARS = (
    "SPACES_ENDPOINT_URL",
    "SPACES_REGION",
    "SPACES_BUCKET",
    "SPACES_ACCESS_KEY_ID",
    "SPACES_SECRET_ACCESS_KEY",
)
DEFAULT_SPACES_PREFIX = "optibot-job"


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def timestamped_run_name(summary: dict[str, Any]) -> str:
    return summary["run_started_at"].replace(":", "").replace(".", "-")


@dataclass
class LocalSyncStore:
    state_path: Path
    runs_dir: Path

    def read_state(self, default: dict[str, Any]) -> dict[str, Any]:
        if not self.state_path.exists():
            return default
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def write_state(self, payload: dict[str, Any]) -> None:
        write_json_file(self.state_path, payload)

    def write_run_artifacts(self, summary: dict[str, Any]) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        write_json_file(self.runs_dir / "latest.json", summary)
        write_json_file(self.runs_dir / f"{timestamped_run_name(summary)}.json", summary)


@dataclass
class SpacesSyncStore:
    local_store: LocalSyncStore
    bucket: str
    prefix: str
    client: Any

    @property
    def state_key(self) -> str:
        return self.object_key("job_state/sync_state.json")

    def object_key(self, suffix: str) -> str:
        clean_prefix = self.prefix.strip("/")
        clean_suffix = suffix.lstrip("/")
        if not clean_prefix:
            return clean_suffix
        return f"{clean_prefix}/{clean_suffix}"

    def read_state(self, default: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.state_key)
        except Exception as exc:
            if is_missing_object_error(exc):
                return default
            raise

        body = response["Body"].read().decode("utf-8")
        return json.loads(body)

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    def write_state(self, payload: dict[str, Any]) -> None:
        self.local_store.write_state(payload)
        self.put_json(self.state_key, payload)

    def write_run_artifacts(self, summary: dict[str, Any]) -> None:
        self.local_store.write_run_artifacts(summary)
        self.put_json(self.object_key("job_runs/latest.json"), summary)
        self.put_json(
            self.object_key(f"job_runs/{timestamped_run_name(summary)}.json"),
            summary,
        )


def is_missing_object_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False

    error = response.get("Error") or {}
    code = str(error.get("Code") or "")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NoSuchKey", "404", "NotFound"} or status == 404


def any_spaces_env_present() -> bool:
    return any(os.environ.get(name) for name in REQUIRED_SPACES_ENV_VARS + ("SPACES_PREFIX",))


def missing_spaces_env_vars() -> list[str]:
    return [name for name in REQUIRED_SPACES_ENV_VARS if not os.environ.get(name)]


def build_spaces_client() -> Any:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "boto3 is required for DigitalOcean Spaces persistence. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return boto3.client(
        "s3",
        endpoint_url=os.environ["SPACES_ENDPOINT_URL"],
        region_name=os.environ["SPACES_REGION"],
        aws_access_key_id=os.environ["SPACES_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SPACES_SECRET_ACCESS_KEY"],
    )


def sync_store_from_env(state_path: Path, runs_dir: Path) -> LocalSyncStore | SpacesSyncStore:
    local_store = LocalSyncStore(state_path=state_path, runs_dir=runs_dir)
    if not any_spaces_env_present():
        print("Using local sync state storage.", flush=True)
        return local_store

    missing = missing_spaces_env_vars()
    if missing:
        raise RuntimeError(
            "DigitalOcean Spaces persistence is partially configured. "
            "Missing: " + ", ".join(missing)
        )

    bucket = os.environ["SPACES_BUCKET"]
    prefix = os.environ.get("SPACES_PREFIX") or DEFAULT_SPACES_PREFIX
    print(f"Using Spaces sync state storage: bucket={bucket}, prefix={prefix}", flush=True)
    return SpacesSyncStore(
        local_store=local_store,
        bucket=bucket,
        prefix=prefix,
        client=build_spaces_client(),
    )
