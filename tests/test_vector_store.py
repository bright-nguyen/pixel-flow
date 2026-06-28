import io
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from optibot_scraper.vector_store import (
    build_report,
    chunking_strategy,
    file_batch_error,
    estimate_chunk_count,
    estimate_files,
    main,
    markdown_files,
    prepare_chunks,
    select_markdown_files,
    strip_image_links,
    validate_chunking,
)


class VectorStoreTests(unittest.TestCase):
    def test_estimate_chunk_count_uses_static_chunk_stride(self):
        self.assertEqual(estimate_chunk_count(0, chunk_size=800, chunk_overlap=400), 0)
        self.assertEqual(estimate_chunk_count(800, chunk_size=800, chunk_overlap=400), 1)
        self.assertEqual(estimate_chunk_count(801, chunk_size=800, chunk_overlap=400), 2)
        self.assertEqual(estimate_chunk_count(1201, chunk_size=800, chunk_overlap=400), 3)

    def test_validate_chunking_rejects_overlap_above_half_chunk_size(self):
        with self.assertRaises(ValueError):
            validate_chunking(chunk_size=800, chunk_overlap=401)

    def test_markdown_files_ignores_non_markdown_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.md").write_text("b", encoding="utf-8")
            (root / "a.md").write_text("a", encoding="utf-8")
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertEqual(
                [path.name for path in markdown_files(root)],
                ["a.md", "b.md"],
            )

    def test_build_report_summarizes_local_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "article.md"
            path.write_text("# Article\n\nArticle URL: https://example.com", encoding="utf-8")
            estimates = estimate_files([path], chunk_size=800, chunk_overlap=400)

            report = build_report(
                estimates,
                chunk_size=800,
                chunk_overlap=400,
                dry_run=True,
            )

            self.assertTrue(report["dry_run"])
            self.assertEqual(report["local_file_count"], 1)
            self.assertEqual(report["estimated_chunk_count"], 1)
            self.assertEqual(report["chunking_strategy"], chunking_strategy(800, 400))

    def test_file_batch_error_reports_failed_indexing(self):
        batch = SimpleNamespace(
            status="failed",
            file_counts=SimpleNamespace(total=3, completed=2, failed=1, cancelled=0),
        )

        self.assertIn("failed=1", file_batch_error(batch))

    def test_upload_main_returns_non_zero_when_batch_indexing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "chunks"
            input_dir.mkdir()
            (input_dir / "article.md").write_text("# Article", encoding="utf-8")
            report_path = root / "report.json"
            failed_batch = SimpleNamespace(
                id="batch_failed",
                status="failed",
                file_counts=SimpleNamespace(total=1, completed=0, failed=1, cancelled=0),
            )

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                with patch("sys.argv", [
                    "main.py",
                    "--input-dir",
                    str(input_dir),
                    "--report-path",
                    str(report_path),
                    "--min-files",
                    "1",
                ]):
                    with patch(
                        "optibot_scraper.vector_store.upload_files",
                        return_value=("vs_failed", failed_batch),
                    ):
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(main(), 1)

            self.assertIn(
                "OpenAI vector-store file batch did not complete successfully",
                report_path.read_text(encoding="utf-8"),
            )

    def test_strip_image_links_keeps_semantic_placeholder_only(self):
        markdown = (
            "Before\n\n"
            "![Image: Setup - 1](../images/article-1.png)\n\n"
            "![firefox_noise.png](data:image/png;base64,abc)\n"
            "[Embedded media: Setup - 1](https://www.canva.com/design/demo/view?embed)\n"
        )

        stripped = strip_image_links(markdown)

        self.assertIn("[Image: Setup - 1]", stripped)
        self.assertIn("[Image: firefox_noise.png]", stripped)
        self.assertIn("[Embedded media: Setup - 1]", stripped)
        self.assertNotIn("../images/article-1.png", stripped)
        self.assertNotIn("data:image", stripped)
        self.assertNotIn("canva.com", stripped)

    def test_prepare_chunks_groups_sections_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "articles"
            output_dir = root / "chunks"
            source_dir.mkdir()
            (source_dir / "1-article.md").write_text(
                "# Article\n\n"
                "Article URL: https://example.com/article\n\n"
                "Article ID: 1\n\n"
                "Last Updated: 2026-01-01T00:00:00Z\n\n"
                "---\n\n"
                "Intro text.\n\n"
                "## Setup\n\n"
                "Follow this step.\n\n"
                "![Image: Setup - 1](../images/a.png)\n",
                encoding="utf-8",
            )

            manifest = prepare_chunks(source_dir, output_dir, max_tokens=1000)
            chunk_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(output_dir.glob("*.md"))
            )

            self.assertEqual(manifest["source_file_count"], 1)
            self.assertGreaterEqual(manifest["chunk_file_count"], 1)
            self.assertIn("Article URL: https://example.com/article", chunk_text)
            self.assertIn("Section Path:", chunk_text)
            self.assertIn("[Image: Setup - 1]", chunk_text)
            self.assertNotIn("../images/a.png", chunk_text)
            self.assertTrue((output_dir / "manifest.json").exists())

    def test_select_markdown_files_prefers_query_matches_and_keeps_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(35):
                title = "How to add a YouTube video" if index == 7 else f"Article {index}"
                body = "YouTube video instructions" if index == 7 else "General OptiSigns article"
                (root / f"{index:02d}-article.md").write_text(
                    f"# {title}\n\n"
                    f"Article URL: https://example.com/{index}\n\n"
                    "---\n\n"
                    f"{body}",
                    encoding="utf-8",
                )

            selected = select_markdown_files(
                root,
                query="How do I add a YouTube video?",
                min_sources=30,
            )

            self.assertEqual(len(selected), 30)
            self.assertEqual(selected[0].name, "07-article.md")


if __name__ == "__main__":
    unittest.main()
