# OptiBot Mini-Clone

Mini support bot pipeline for OptiSigns Help Center articles:

1. Scrape public Zendesk articles into clean Markdown.
2. Prepare section-aware RAG chunks and upload them to an OpenAI Vector Store.
3. Run the same scrape/chunk/upload flow as a daily delta-sync job.

## Requirements

- Python 3.12+
- `OPENAI_API_KEY`, either in `.env` or injected as an environment variable.

## Local Run

**Setup**

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

>If you want to use a `.env` file, run command: `cp .env.sample .env` and add your OpenAI API key to the `.env` file.  Otherwise, inject the API key per command: `OPENAI_API_KEY=sk-... python <other commands>`

**Build and upload the knowledge base**:

```bash
python main.py scrape --clean --limit 30
python main.py prepare-chunks
```
> ⚠️ For a faster local smoke test, scrape and chunk only 30 articles. But the Assistant can only retrieve answers from those 30 scraped articles, make sure scraped contain the articles you want to test.

> Remove `--limit 30` to scrape full corpus of articles - this can take a while (~20 minutes for 402 articles).

**Upload the vector store to OpenAI**:

```bash
python main.py upload-vector-store --vector-store-name "OptiBot Support Articles"
```

**Quick sanity check the Assistant**

Follow the take-home instructions: [Quick sanity check](https://docs.google.com/document/d/1V3QXfoGCk6toSs8QFbaKzSp-deuCAoN2COF7Ki_VPQ4/edit?tab=t.0#heading=h.jrz269ofkxd8).

- It should look like this:
![playground screenshot](screen-shot.png)

- Daily job artifacts:
  - [First job artifact](https://sgp1.digitaloceanspaces.com/takehome28062026/optibot-job/job_runs/2026-06-28T070421-080893Z.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=DO00MHYRMZKUZUPAQWEW%2F20260628%2Fsgp1%2Fs3%2Faws4_request&X-Amz-Date=20260628T081331Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c81ab5ead64f5c1dbb35041a899e6cf96911c45ee710cabfbe5e0b9558cdd749): Scraped 402 articles and uploaded 713 chunks.
  - [Latest job artifact](https://sgp1.digitaloceanspaces.com/takehome28062026/optibot-job/job_runs/latest.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=DO00MHYRMZKUZUPAQWEW%2F20260628%2Fsgp1%2Fs3%2Faws4_request&X-Amz-Date=20260628T080607Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=ebfd55d863ac49fa21c9764245e1fa8888f3b8a0fa5d2519e3ee2e20384985fe): The run artifact shows the required job counts (`added`, `updated`, `skipped`, etc.).
  - [Sync state](https://sgp1.digitaloceanspaces.com/takehome28062026/optibot-job/job_state/sync_state.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=DO00MHYRMZKUZUPAQWEW%2F20260628%2Fsgp1%2Fs3%2Faws4_request&X-Amz-Date=20260628T080252Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9da0fe770db917ad74d1cbdb39005bdfa08124fdf7b9d535e2e49e38bb4b07dd): The sync state is persisted separately and stores article hashes plus OpenAI file IDs so later runs can skip unchanged articles instead of re-uploading them.

⚠️ Access links are temporary and expire at 2026-07-05 08:00 UTC.


Run the scheduled-job flow locally without writing to OpenAI:

```bash
python main.py sync --dry-run
```

The sync job prints a JSON summary to stdout and writes a local run artifact at [data/job_runs/latest.json](data/job_runs/latest.json). Use `python main.py upload-vector-store --dry-run` when you only want upload estimates.

## Unit Test

```bash
python -m unittest discover
```
