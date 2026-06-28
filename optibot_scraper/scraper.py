from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by users before setup
    missing = exc.name or "a dependency"
    raise SystemExit(
        f"Missing {missing}. Install dependencies with: pip install -r requirements.txt"
    ) from exc


DEFAULT_BASE_URL = "https://support.optisigns.com"
DEFAULT_LOCALE = "en-us"
DEFAULT_OUTPUT_DIR = Path("data/articles")
DEFAULT_IMAGE_DIR_NAME = "images"
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
IMAGE_EXTENSIONS_BY_MIME = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class Article:
    article_id: int
    title: str
    html_url: str
    updated_at: str
    body: str

    @classmethod
    def from_api(cls, raw: dict) -> "Article | None":
        try:
            article_id = int(raw["id"])
        except (KeyError, TypeError, ValueError):
            return None

        title = (raw.get("title") or raw.get("name") or "").strip()
        html_url = (raw.get("html_url") or "").strip()
        body = raw.get("body") or ""
        if not title or not html_url or not body:
            return None

        return cls(
            article_id=article_id,
            title=title,
            html_url=html_url,
            updated_at=(raw.get("updated_at") or "").strip(),
            body=body,
        )


@dataclass(frozen=True)
class FetchResult:
    articles: list[Article]
    skipped_count: int


def fetch_articles(
    base_url: str = DEFAULT_BASE_URL,
    locale: str = DEFAULT_LOCALE,
    limit: int | None = None,
    per_page: int = 100,
) -> FetchResult:
    """Fetch public Help Center articles through Zendesk pagination."""
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "optibot-takehome-scraper/1.0",
        }
    )

    url: str | None = (
        f"{base_url.rstrip('/')}/api/v2/help_center/{locale}/articles.json"
    )
    params: dict | None = {"per_page": min(max(per_page, 1), 100)}
    articles: list[Article] = []
    skipped_count = 0

    while url:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for raw in payload.get("articles", []):
            article = Article.from_api(raw)
            if article is None:
                skipped_count += 1
                continue

            articles.append(article)
            if limit is not None and len(articles) >= limit:
                return FetchResult(articles=articles, skipped_count=skipped_count)

        url = payload.get("next_page")
        params = None

    return FetchResult(articles=articles, skipped_count=skipped_count)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_value).strip("-").lower()
    return slug or "article"


def heading_slug(value: str) -> str:
    return slugify(value)


def filename_for_article(article: Article) -> str:
    path_part = unquote(urlparse(article.html_url).path.rstrip("/").split("/")[-1])
    base = slugify(path_part or article.title)
    article_id = str(article.article_id)
    if base != article_id and not base.startswith(f"{article_id}-"):
        base = f"{article_id}-{base}"
    return f"{base}.md"


def build_filename_map(articles: Iterable[Article]) -> dict[int, str]:
    return {article.article_id: filename_for_article(article) for article in articles}


def unique_heading_slugs(soup: BeautifulSoup) -> dict[int, str]:
    slugs_by_heading_id = {}
    slug_counts = {}

    for heading in soup.find_all(HEADING_TAGS):
        base_slug = heading_slug(heading.get_text(" ", strip=True))
        count = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = count + 1
        slug = base_slug if count == 0 else f"{base_slug}-{count}"
        slugs_by_heading_id[id(heading)] = slug

    return slugs_by_heading_id


def build_article_fragment_map(article: Article) -> dict[str, str]:
    soup = BeautifulSoup(article.body, "html.parser")
    heading_slugs = unique_heading_slugs(soup)
    fragment_map = {}

    for heading in soup.find_all(HEADING_TAGS):
        heading_target = (heading.get("id") or heading.get("name") or "").strip()
        if heading_target:
            fragment_map[heading_target] = heading_slugs[id(heading)]

    for anchor in soup.find_all("a"):
        if anchor.get("href"):
            continue

        target = (anchor.get("name") or anchor.get("id") or "").strip()
        if not target:
            continue

        heading = anchor.find_parent(HEADING_TAGS) or anchor.find_next(HEADING_TAGS)
        if heading is None:
            continue

        fragment_map[target] = heading_slugs[id(heading)]

    return fragment_map


def build_fragment_map(articles: Iterable[Article]) -> dict[int, dict[str, str]]:
    return {
        article.article_id: build_article_fragment_map(article)
        for article in articles
    }


def article_id_from_href(href: str) -> int | None:
    parsed = urlparse(href)
    match = re.search(r"/articles/(\d+)", parsed.path)
    if not match:
        return None
    return int(match.group(1))


def is_opaque_fragment(fragment: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,3}", fragment))


def resolve_fragment(
    article_id: int,
    fragment: str,
    link_text: str,
    fragment_by_article_id: dict[int, dict[str, str]],
) -> str:
    mapped_fragment = fragment_by_article_id.get(article_id, {}).get(fragment)
    if mapped_fragment:
        return mapped_fragment

    if is_opaque_fragment(fragment) and link_text:
        return heading_slug(link_text)

    return fragment


def rewrite_internal_links(
    soup: BeautifulSoup,
    filename_by_id: dict[int, str],
    fragment_by_article_id: dict[int, dict[str, str]],
    current_article_id: int | None = None,
) -> None:
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("mailto:", "tel:")):
            continue

        if href.startswith("#") and current_article_id is not None:
            fragment = href[1:]
            mapped_fragment = resolve_fragment(
                current_article_id,
                fragment,
                anchor.get_text(" ", strip=True),
                fragment_by_article_id,
            )
            anchor["href"] = f"#{mapped_fragment}"
            continue

        article_id = article_id_from_href(href)
        if article_id is None or article_id not in filename_by_id:
            continue

        fragment = urlparse(href).fragment
        mapped_fragment = resolve_fragment(
            article_id,
            fragment,
            anchor.get_text(" ", strip=True),
            fragment_by_article_id,
        )
        if current_article_id == article_id and fragment:
            anchor["href"] = f"#{mapped_fragment}"
            continue

        relative_href = f"./{filename_by_id[article_id]}"
        if fragment:
            relative_href = f"{relative_href}#{mapped_fragment}"
        anchor["href"] = relative_href


def heading_contains_only_image(heading) -> bool:
    return (
        heading.name in HEADING_TAGS
        and heading.find("img") is not None
        and not heading.get_text(" ", strip=True)
    )


def replace_image_only_headings(soup: BeautifulSoup) -> None:
    for heading in soup.find_all(HEADING_TAGS):
        if not heading_contains_only_image(heading):
            continue

        paragraph = soup.new_tag("p")
        for child in list(heading.contents):
            paragraph.append(child.extract())
        heading.replace_with(paragraph)


def attachment_id_from_src(src: str) -> str | None:
    parsed = urlparse(src)
    match = re.search(r"/article_attachments/(\d+)", parsed.path)
    if match:
        return match.group(1)
    return None


def decode_data_image(
    src: str,
    article_id: int,
    image_dir: Path | None,
) -> str | None:
    if image_dir is None or not src.startswith("data:image/"):
        return None

    match = re.match(r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.+)$", src, re.S)
    if not match:
        return None

    mime_type, encoded = match.groups()
    extension = IMAGE_EXTENSIONS_BY_MIME.get(mime_type.lower(), ".img")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None

    digest = hashlib.sha256(data).hexdigest()[:12]
    filename = f"article-{article_id}-embedded-image-{digest}{extension}"
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / filename
    if not path.exists():
        path.write_bytes(data)
    return filename


def local_image_href(image_dir: Path, markdown_dir: Path, filename: str) -> str:
    return Path("..", image_dir.name, filename).as_posix()


def normalize_images(
    soup: BeautifulSoup,
    article: Article,
    image_dir: Path | None = None,
    markdown_dir: Path | None = None,
) -> None:
    current_heading = article.title
    image_counts_by_heading: dict[str, int] = {}

    replace_image_only_headings(soup)

    for element in soup.find_all(list(HEADING_TAGS) + ["img"]):
        if element.name in HEADING_TAGS:
            heading_text = element.get_text(" ", strip=True)
            if heading_text:
                current_heading = heading_text
            continue

        count = image_counts_by_heading.get(current_heading, 0) + 1
        image_counts_by_heading[current_heading] = count
        element["alt"] = f"Image: {current_heading} - {count}"

        src = (element.get("src") or "").strip()
        if src.startswith("data:image/"):
            filename = decode_data_image(src, article.article_id, image_dir)
            if filename and image_dir is not None and markdown_dir is not None:
                element["src"] = local_image_href(image_dir, markdown_dir, filename)
                element["data-optibot-image-file"] = filename
            else:
                element.decompose()
            continue

        attachment_id = attachment_id_from_src(src)
        if attachment_id:
            element["data-optibot-attachment-id"] = attachment_id


def escape_markdown_link_text(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]")


def escape_markdown_link_target(value: str) -> str:
    return value.replace(")", "%29")


def markdown_image_text(alt_text: str, src: str) -> str:
    return f"![{escape_markdown_link_text(alt_text)}]({escape_markdown_link_target(src)})"


def markdown_link_text(label: str, href: str) -> str:
    return f"[{escape_markdown_link_text(label)}]({escape_markdown_link_target(href)})"


def replace_images_with_markdown_text(soup: BeautifulSoup) -> None:
    for image in soup.find_all("img"):
        src = (image.get("src") or "").strip()
        if not src:
            image.decompose()
            continue

        alt_text = (image.get("alt") or "Image").strip()
        image.replace_with(soup.new_string(markdown_image_text(alt_text, src)))


def normalize_iframes(soup: BeautifulSoup, article: Article) -> None:
    current_heading = article.title
    media_counts_by_heading: dict[str, int] = {}

    for element in soup.find_all(list(HEADING_TAGS) + ["iframe"]):
        if element.name in HEADING_TAGS:
            heading_text = element.get_text(" ", strip=True)
            if heading_text:
                current_heading = heading_text
            continue

        src = (element.get("src") or "").strip()
        if not src:
            element.decompose()
            continue

        count = media_counts_by_heading.get(current_heading, 0) + 1
        media_counts_by_heading[current_heading] = count
        label = f"Embedded media: {current_heading} - {count}"
        element.replace_with(soup.new_string(markdown_link_text(label, src)))


def clean_article_html(
    article: Article,
    filename_by_id: dict[int, str],
    fragment_by_article_id: dict[int, dict[str, str]],
    image_dir: Path | None = None,
    markdown_dir: Path | None = None,
    current_article_id: int | None = None,
) -> str:
    soup = BeautifulSoup(article.body, "html.parser")

    for element in soup(["script", "style", "noscript", "form"]):
        element.decompose()

    normalize_images(
        soup,
        article,
        image_dir=image_dir,
        markdown_dir=markdown_dir,
    )
    normalize_iframes(soup, article)
    replace_images_with_markdown_text(soup)
    rewrite_internal_links(
        soup,
        filename_by_id,
        fragment_by_article_id,
        current_article_id=current_article_id,
    )
    return str(soup)


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = markdown.replace("\\_", "_")
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def fence_iframe_snippets(markdown: str) -> str:
    lines = markdown.splitlines()
    fenced_lines = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            fenced_lines.append(line)
            continue

        if not in_code_fence and "<iframe" in stripped and "</iframe>" in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            fenced_lines.append(f"{indent}```html")
            fenced_lines.append(line)
            fenced_lines.append(f"{indent}```")
            continue

        fenced_lines.append(line)

    return "\n".join(fenced_lines)


def article_to_markdown(
    article: Article,
    filename_by_id: dict[int, str],
    fragment_by_article_id: dict[int, dict[str, str]] | None = None,
    image_dir: Path | None = None,
    markdown_dir: Path | None = None,
) -> str:
    fragment_by_article_id = fragment_by_article_id or {}
    markdown_dir = markdown_dir or DEFAULT_OUTPUT_DIR
    cleaned_html = clean_article_html(
        article,
        filename_by_id,
        fragment_by_article_id,
        image_dir=image_dir,
        markdown_dir=markdown_dir,
        current_article_id=article.article_id,
    )
    body_markdown = markdownify(
        cleaned_html,
        bullets="-",
        heading_style="ATX",
        strip=["script", "style", "noscript"],
    )
    body_markdown = fence_iframe_snippets(body_markdown)
    body_markdown = normalize_markdown(body_markdown)

    metadata = [
        f"# {article.title}",
        "",
        f"Article URL: {article.html_url}",
        "",
        f"Article ID: {article.article_id}",
        "",
        f"Last Updated: {article.updated_at}",
        "",
        "---",
        "",
    ]
    return "\n".join(metadata + [body_markdown, ""])


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_articles(
    articles: list[Article],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    clean: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir.parent / DEFAULT_IMAGE_DIR_NAME
    if clean:
        for existing in output_dir.glob("*.md"):
            existing.unlink()

    filename_by_id = build_filename_map(articles)
    fragment_by_article_id = build_fragment_map(articles)
    manifest_articles = []

    for article in articles:
        filename = filename_by_id[article.article_id]
        markdown = article_to_markdown(
            article,
            filename_by_id,
            fragment_by_article_id,
            image_dir=image_dir,
            markdown_dir=output_dir,
        )
        path = output_dir / filename
        path.write_text(markdown, encoding="utf-8")
        manifest_articles.append(
            {
                "id": article.article_id,
                "title": article.title,
                "url": article.html_url,
                "updated_at": article.updated_at,
                "filename": filename,
                "sha256": content_hash(markdown),
            }
        )

    manifest = {
        "article_count": len(manifest_articles),
        "articles": manifest_articles,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape OptiSigns Zendesk articles into Markdown."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--locale", default=DEFAULT_LOCALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-articles", type=int, default=30)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing Markdown files from the output directory first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = fetch_articles(
        base_url=args.base_url,
        locale=args.locale,
        limit=args.limit,
        per_page=args.per_page,
    )
    articles = result.articles

    if len(articles) < args.min_articles:
        print(
            f"Expected at least {args.min_articles} articles, fetched {len(articles)}.",
        )
        return 1

    manifest = write_articles(articles, output_dir=args.output_dir, clean=args.clean)
    print(
        f"Wrote {manifest['article_count']} Markdown files "
        f"to {args.output_dir} plus manifest.json."
    )
    print(f"Skipped {result.skipped_count} malformed or empty article records.")
    return 0
