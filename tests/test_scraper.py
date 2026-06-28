import unittest
import tempfile
from pathlib import Path

from optibot_scraper.scraper import (
    Article,
    article_to_markdown,
    build_fragment_map,
    build_filename_map,
    filename_for_article,
)


class ScraperTests(unittest.TestCase):
    def test_filename_uses_article_id_and_slug(self):
        article = Article(
            article_id=12345,
            title="How to Add a YouTube Video",
            html_url="https://support.optisigns.com/hc/en-us/articles/12345-How-to-Add-a-YouTube-Video",
            updated_at="2026-01-01T00:00:00Z",
            body="<p>Hello</p>",
        )

        self.assertEqual(
            filename_for_article(article),
            "12345-how-to-add-a-youtube-video.md",
        )

    def test_article_markdown_rewrites_internal_links_to_relative_files(self):
        source = Article(
            article_id=1,
            title="Source Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/1-Source-Article",
            updated_at="2026-01-01T00:00:00Z",
            body='<p>See <a href="https://support.optisigns.com/hc/en-us/articles/2-Target-Article#steps">target</a>.</p>',
        )
        target = Article(
            article_id=2,
            title="Target Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/2-Target-Article",
            updated_at="2026-01-01T00:00:00Z",
            body="<p>Target</p>",
        )

        articles = [source, target]
        markdown = article_to_markdown(
            source,
            build_filename_map(articles),
            build_fragment_map(articles),
        )

        self.assertIn("[target](./2-target-article.md#steps)", markdown)
        self.assertIn("Article URL: https://support.optisigns.com/hc/en-us/articles/1-Source-Article", markdown)

    def test_same_page_anchor_links_use_semantic_heading_slugs(self):
        article = Article(
            article_id=10,
            title="Anchors",
            html_url="https://support.optisigns.com/hc/en-us/articles/10-Anchors",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<ul><li><a href="#WhatYouNeed">What You Need</a></li></ul>'
                '<p><a name="WhatYouNeed"></a></p>'
                "<h2>What You Need</h2>"
            ),
        )

        markdown = article_to_markdown(
            article,
            build_filename_map([article]),
            build_fragment_map([article]),
        )

        self.assertIn("[What You Need](#what-you-need)", markdown)
        self.assertNotIn('<a id="WhatYouNeed"></a>', markdown)
        self.assertIn("## What You Need", markdown)

    def test_full_self_link_with_fragment_becomes_same_page_anchor(self):
        article = Article(
            article_id=11,
            title="Self Link",
            html_url="https://support.optisigns.com/hc/en-us/articles/11-Self-Link",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<p><a href="https://support.optisigns.com/hc/en-us/articles/11-Self-Link#Setup">'
                "Setup section</a></p>"
                '<p><a name="Setup"></a></p><h2>Setup</h2>'
            ),
        )

        markdown = article_to_markdown(
            article,
            build_filename_map([article]),
            build_fragment_map([article]),
        )

        self.assertIn("[Setup section](#setup)", markdown)
        self.assertNotIn("./11-self-link.md#Setup", markdown)

    def test_numeric_anchor_links_use_target_heading_slug(self):
        article = Article(
            article_id=13,
            title="Numeric",
            html_url="https://support.optisigns.com/hc/en-us/articles/13-Numeric",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<ul><li><a href="#1">Option 1</a></li></ul>'
                '<p><a name="1"></a></p>'
                "<h2>Option 1: Install on an SSSP Display That is Already in Use:</h2>"
            ),
        )

        markdown = article_to_markdown(
            article,
            build_filename_map([article]),
            build_fragment_map([article]),
        )

        self.assertIn(
            "[Option 1](#option-1-install-on-an-sssp-display-that-is-already-in-use)",
            markdown,
        )
        self.assertNotIn("](#1)", markdown)

    def test_cross_article_anchor_links_use_target_heading_slug(self):
        source = Article(
            article_id=14,
            title="Source",
            html_url="https://support.optisigns.com/hc/en-us/articles/14-Source",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<p><a href="https://support.optisigns.com/hc/en-us/articles/15-Target#2">'
                "Option 2</a></p>"
            ),
        )
        target = Article(
            article_id=15,
            title="Target",
            html_url="https://support.optisigns.com/hc/en-us/articles/15-Target",
            updated_at="2026-01-01T00:00:00Z",
            body='<p><a name="2"></a></p><h2>Option 2: Configure Display</h2>',
        )
        articles = [source, target]

        markdown = article_to_markdown(
            source,
            build_filename_map(articles),
            build_fragment_map(articles),
        )

        self.assertIn(
            "[Option 2](./15-target.md#option-2-configure-display)",
            markdown,
        )

    def test_unmapped_opaque_anchor_uses_link_text_slug(self):
        article = Article(
            article_id=16,
            title="Paragraph Anchor",
            html_url="https://support.optisigns.com/hc/en-us/articles/16-Paragraph-Anchor",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<ul><li><a href="#0">Android OS device</a></li></ul>'
                '<p><a name="0"></a></p>'
                "<p>For Android OS devices such as the OptiSigns Android Stick.</p>"
            ),
        )

        markdown = article_to_markdown(
            article,
            build_filename_map([article]),
            build_fragment_map([article]),
        )

        self.assertIn("[Android OS device](#android-os-device)", markdown)
        self.assertNotIn("](#0)", markdown)

    def test_external_and_mail_links_are_preserved(self):
        article = Article(
            article_id=3,
            title="Links",
            html_url="https://support.optisigns.com/hc/en-us/articles/3-Links",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                '<p><a href="https://example.com/page">external</a> '
                '<a href="mailto:support@optisigns.com">email</a></p>'
            ),
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertIn("[external](https://example.com/page)", markdown)
        self.assertIn("[email](mailto:support@optisigns.com)", markdown)

    def test_iframe_snippets_are_wrapped_in_code_fences(self):
        article = Article(
            article_id=12,
            title="Iframe",
            html_url="https://support.optisigns.com/hc/en-us/articles/12-Iframe",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                "<ol><li>Paste this code:<br><br>"
                '&lt;iframe src="https://example.com"&gt;&lt;/iframe&gt;'
                "</li></ol>"
            ),
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertIn(
            '```html\n   <iframe src="https://example.com"></iframe>\n   ```',
            markdown,
        )

    def test_markdown_includes_required_metadata(self):
        article = Article(
            article_id=4,
            title="Metadata",
            html_url="https://support.optisigns.com/hc/en-us/articles/4-Metadata",
            updated_at="2026-01-01T00:00:00Z",
            body="<h2>Body Heading</h2><p>Body text</p>",
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertTrue(markdown.startswith("# Metadata\n\n"))
        self.assertIn(
            "Article URL: https://support.optisigns.com/hc/en-us/articles/4-Metadata\n\n"
            "Article ID: 4\n\n"
            "Last Updated: 2026-01-01T00:00:00Z",
            markdown,
        )
        self.assertIn("## Body Heading", markdown)

    def test_image_only_heading_becomes_contextual_image_not_markdown_heading(self):
        article = Article(
            article_id=20,
            title="Image Article",
            html_url="https://support.optisigns.com/hc/en-us/articles/20-Image-Article",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                "<h2>Option 1: With a Design</h2>"
                '<h3><img src="https://support.optisigns.com/hc/article_attachments/123" '
                'alt="firefox_Roe3CPM3YA.png"></h3>'
            ),
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertIn(
            "![Image: Option 1: With a Design - 1]"
            "(https://support.optisigns.com/hc/article_attachments/123)",
            markdown,
        )
        self.assertNotIn("### firefox_Roe3CPM3YA.png", markdown)

    def test_base64_image_is_decoded_to_local_file(self):
        article = Article(
            article_id=21,
            title="Inline Image",
            html_url="https://support.optisigns.com/hc/en-us/articles/21-Inline-Image",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                "<h2>Setup</h2>"
                '<p><img src="data:image/png;base64,iVBORw0KGgo=" '
                'alt="firefox_inline.png"></p>'
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            markdown = article_to_markdown(
                article,
                build_filename_map([article]),
                image_dir=root / "images",
                markdown_dir=root / "articles",
            )

            self.assertIn("![Image: Setup - 1](../images/article-21-embedded-image-", markdown)
            self.assertNotIn("data:image/png;base64", markdown)
            self.assertEqual(len(list((root / "images").glob("*.png"))), 1)

    def test_table_images_keep_markdown_image_links(self):
        article = Article(
            article_id=22,
            title="Table Images",
            html_url="https://support.optisigns.com/hc/en-us/articles/22-Table-Images",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                "<h2>Creating a Schedule</h2>"
                "<table><tr><td><strong>OptiSigns 1.0</strong></td></tr>"
                '<tr><td><img src="https://support.optisigns.com/hc/article_attachments/123"></td></tr></table>'
            ),
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertIn(
            "![Image: Creating a Schedule - 1]"
            "(https://support.optisigns.com/hc/article_attachments/123)",
            markdown,
        )

    def test_iframe_embed_becomes_markdown_link(self):
        article = Article(
            article_id=23,
            title="Embeds",
            html_url="https://support.optisigns.com/hc/en-us/articles/23-Embeds",
            updated_at="2026-01-01T00:00:00Z",
            body=(
                "<h2>Creating a Schedule</h2>"
                "<table><tr><td>"
                '<iframe src="https://www.canva.com/design/demo/view?embed"></iframe>'
                "</td></tr></table>"
            ),
        )

        markdown = article_to_markdown(article, build_filename_map([article]))

        self.assertIn(
            "[Embedded media: Creating a Schedule - 1]"
            "(https://www.canva.com/design/demo/view?embed)",
            markdown,
        )

    def test_malformed_api_records_return_none(self):
        self.assertIsNone(Article.from_api({"title": "Missing ID"}))
        self.assertIsNone(Article.from_api({"id": 5, "title": "", "html_url": "u", "body": "b"}))


if __name__ == "__main__":
    unittest.main()
