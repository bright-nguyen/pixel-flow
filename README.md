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

- Daily job logs link:

Run the scheduled-job flow locally without writing to OpenAI:

```bash
python main.py sync --dry-run
```

The sync job prints a JSON summary to stdout and writes a local run artifact at [data/job_runs/latest.json](data/job_runs/latest.json). Use `python main.py upload-vector-store --dry-run` when you only want upload estimates.

## Unit Test

```bash
python -m unittest discover
```
