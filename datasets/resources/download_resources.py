from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime in PDF extraction
    PdfReader = None


BASE_DIR = Path(__file__).resolve().parent
LINKS_PATH = BASE_DIR / "useful_links.txt"
RAW_DIR = BASE_DIR / "raw"
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_PATH = BASE_DIR / "index.json"
REPORT_PATH = BASE_DIR / "download_report.md"

USER_AGENT = "AAI-project-resource-downloader/1.0 (+local dataset preparation)"
TIMEOUT_SECONDS = 40


def slugify(value: str, max_length: int = 70) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return (slug or "resource")[:max_length].strip("-")


def parse_useful_links(path: Path) -> list[dict]:
    entries = []
    current_label = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("http://", "https://")):
            if current_label is None:
                raise ValueError(f"URL without a preceding label: {line}")
            entries.append({"label": current_label, "url": line})
        elif line.endswith(":"):
            current_label = line[:-1].strip()

    return entries


def unique_resources(entries: Iterable[dict]) -> list[dict]:
    by_url: dict[str, dict] = {}
    for entry in entries:
        resource = by_url.setdefault(
            entry["url"],
            {
                "url": entry["url"],
                "labels": [],
                "source_entry_count": 0,
            },
        )
        resource["source_entry_count"] += 1
        if entry["label"] not in resource["labels"]:
            resource["labels"].append(entry["label"])

    resources = list(by_url.values())
    for index, resource in enumerate(resources, start=1):
        primary_label = resource["labels"][0]
        resource["id"] = f"resource_{index:03d}"
        resource["slug"] = slugify(primary_label)
    return resources


def infer_extension(url: str, content_type: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm", ".txt"}:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    if "text/plain" in content_type:
        return ".txt"
    return ".bin"


def raw_filename(resource: dict, extension: str) -> str:
    return f"{resource['id']}_{resource['slug']}{extension}"


def document_filename(resource: dict) -> str:
    return f"{resource['id']}_{resource['slug']}.md"


def download_url(url: str) -> tuple[bytes, str, str, int, str]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        return response.content, content_type, response.url, response.status_code, "direct"
    except requests.RequestException:
        reader_url = f"https://r.jina.ai/http://r.jina.ai/http://{url}"
        response = requests.get(
            reader_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/plain"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        return response.content, content_type or "text/plain", url, response.status_code, "jina_reader"


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def html_to_markdown(content: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    title = clean_text(soup.title.get_text(" ")) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    parts = []

    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_text(element.get_text(" "))
        if not text:
            continue
        if element.name == "h1":
            parts.append(f"# {text}")
        elif element.name == "h2":
            parts.append(f"## {text}")
        elif element.name in {"h3", "h4"}:
            parts.append(f"### {text}")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)

    markdown = "\n\n".join(parts)
    return title, markdown


def pdf_to_markdown(path: Path) -> tuple[str, str]:
    if PdfReader is None:
        raise RuntimeError("pypdf is required for PDF extraction. Install it with `python -m pip install pypdf`.")

    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"## Page {page_number}\n\n{text}")

    return path.stem, "\n\n".join(pages)


def text_to_markdown(content: bytes) -> tuple[str, str]:
    text = content.decode("utf-8", errors="replace").strip()
    return "", text


def metadata_header(record: dict) -> str:
    metadata = {
        "id": record["id"],
        "source_url": record["url"],
        "final_url": record.get("final_url"),
        "labels": record["labels"],
        "content_type": record.get("content_type"),
        "retrieved_at": record["retrieved_at"],
    }
    return "---\n" + json.dumps(metadata, indent=2) + "\n---\n\n"


def write_document(record: dict, title: str, body: str) -> None:
    heading = f"# {title.strip()}\n\n" if title.strip() and not body.lstrip().startswith("# ") else ""
    document_path = BASE_DIR / record["document_file"]
    document_path.write_text(metadata_header(record) + heading + body.strip() + "\n", encoding="utf-8")


def process_resource(resource: dict, retrieved_at: str) -> dict:
    record = {
        "id": resource["id"],
        "url": resource["url"],
        "labels": resource["labels"],
        "source_entry_count": resource["source_entry_count"],
        "retrieved_at": retrieved_at,
        "status": "pending",
    }

    try:
        content, content_type, final_url, status_code, retrieval_method = download_url(resource["url"])
        extension = infer_extension(final_url, content_type)
        raw_file = raw_filename(resource, extension)
        document_file = document_filename(resource)
        raw_path = RAW_DIR / raw_file

        raw_path.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()

        record.update(
            {
                "status_code": status_code,
                "content_type": content_type,
                "final_url": final_url,
                "retrieval_method": retrieval_method,
                "raw_file": str(raw_path.relative_to(BASE_DIR)),
                "document_file": str((DOCUMENTS_DIR / document_file).relative_to(BASE_DIR)),
                "sha256": checksum,
                "bytes": len(content),
            }
        )

        if extension == ".pdf" or "pdf" in content_type:
            title, body = pdf_to_markdown(raw_path)
        elif "html" in content_type or extension in {".html", ".htm"}:
            title, body = html_to_markdown(content)
        elif "text" in content_type or extension == ".txt":
            title, body = text_to_markdown(content)
        else:
            raise RuntimeError(f"Unsupported content type for extraction: {content_type or 'unknown'}")

        if not body.strip():
            raise RuntimeError("Extraction produced no text.")

        record["title"] = title
        record["extracted_characters"] = len(body)
        record["status"] = "success"
        write_document(record, title, body)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def write_report(records: list[dict], total_entries: int) -> None:
    successes = [record for record in records if record["status"] == "success"]
    failures = [record for record in records if record["status"] != "success"]
    duplicate_entries = sum(record["source_entry_count"] - 1 for record in records)

    lines = [
        "# Resource Download Report",
        "",
        f"- Source entries: {total_entries}",
        f"- Unique URLs: {len(records)}",
        f"- Duplicate entries collapsed: {duplicate_entries}",
        f"- Successful documents: {len(successes)}",
        f"- Failed resources: {len(failures)}",
        "",
        "## Failures",
        "",
    ]

    if failures:
        for record in failures:
            labels = ", ".join(record["labels"])
            lines.append(f"- `{record['id']}` {labels}: {record['url']}")
            lines.append(f"  - {record.get('error', 'Unknown error')}")
    else:
        lines.append("No failures.")

    lines.extend(["", "## Documents", ""])
    for record in records:
        labels = ", ".join(record["labels"])
        status = record["status"]
        document = record.get("document_file", "")
        lines.append(f"- `{record['id']}` [{status}] {labels} -> `{document}`")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    entries = parse_useful_links(LINKS_PATH)
    resources = unique_resources(entries)
    retrieved_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()

    records = []
    for resource in resources:
        print(f"{resource['id']} {resource['url']}", flush=True)
        record = process_resource(resource, retrieved_at)
        print(f"  {record['status']}", flush=True)
        if record["status"] == "failed":
            print(f"  {record['error']}", flush=True)
        records.append(record)

    INDEX_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_report(records, total_entries=len(entries))

    success_count = sum(record["status"] == "success" for record in records)
    print(f"\nDone: {success_count}/{len(records)} resources downloaded and extracted.")
    print(f"Index: {INDEX_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
