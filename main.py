from __future__ import annotations

import sys

from optibot_scraper import scraper
from optibot_scraper import sync_job
from optibot_scraper import vector_store


def print_help() -> None:
    print(
        "Usage: python main.py <command> [options]\n\n"
        "Commands:\n"
        "  scrape                Scrape OptiSigns Help Center articles to Markdown\n"
        "  prepare-chunks        Prepare section-aware Markdown chunks for RAG\n"
        "  upload-vector-store   Upload prepared chunks to an OpenAI Vector Store\n"
        "  sync                  Run the daily scrape/chunk/vector-store sync job\n\n"
        "Run `python main.py <command> --help` for command-specific options."
    )


def main() -> int:
    if len(sys.argv) <= 1:
        return scraper.main()

    command = sys.argv[1]
    if command in {"-h", "--help", "help"}:
        print_help()
        return 0

    if command == "sync":
        sys.argv.pop(1)
        return sync_job.main()

    if command == "scrape":
        sys.argv.pop(1)
        return scraper.main()

    if command == "prepare-chunks":
        sys.argv.pop(1)
        return vector_store.prepare_chunks_main()

    if command == "upload-vector-store":
        sys.argv.pop(1)
        return vector_store.main()

    return scraper.main()


if __name__ == "__main__":
    raise SystemExit(main())
