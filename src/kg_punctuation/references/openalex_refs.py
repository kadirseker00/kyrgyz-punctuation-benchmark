from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
USER_AGENT = "KG-Punctuation-Research/0.1 (mailto:example@example.com)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich OpenAlex CSV exports with OA/PDF metadata.")
    parser.add_argument("--input-dir", type=Path, default=Path("paper/refs"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/refs/processed"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("paper/refs/pdfs"))
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = load_export_rows(args.input_dir)
    enriched = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row['Title'][:80]}")
        metadata = lookup_openalex(row)
        merged = {**row, **metadata}
        if args.download_pdfs and merged.get("pdf_url"):
            merged["pdf_file"] = download_pdf(merged["pdf_url"], args.pdf_dir, merged)
        enriched.append(merged)
        time.sleep(args.sleep)

    write_outputs(enriched, args.output_dir)
    print(f"wrote {args.output_dir / 'openalex_enriched_refs.csv'}")
    print(f"wrote {args.output_dir / 'openalex_enriched_refs.md'}")


def load_export_rows(input_dir: Path) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for path in sorted(input_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                title = clean_text(row.get("Title", ""))
                doi = normalize_doi(row.get("DOI", ""))
                key = doi or title.lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "source_file": path.name,
                        "title": title,
                        "authors": clean_text(row.get("Author", "")),
                        "year": clean_text(row.get("Year", "")),
                        "doi": doi,
                        "Title": title,
                    }
                )
    return rows


def lookup_openalex(row: dict[str, str]) -> dict[str, str]:
    try:
        if row.get("doi"):
            work = fetch_json(f"{OPENALEX_WORKS_URL}/https://doi.org/{quote(row['doi'])}")
        else:
            params = urlencode({"search": row["title"], "per-page": 1})
            payload = fetch_json(f"{OPENALEX_WORKS_URL}?{params}")
            results = payload.get("results", [])
            work = results[0] if results else {}
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"lookup_error": str(exc)}

    if not work:
        return {"lookup_error": "not found"}

    best = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    open_access = work.get("open_access") or {}
    locations = work.get("locations") or []
    pdf_url = best.get("pdf_url") or primary.get("pdf_url") or first_pdf_url(locations)
    landing_page_url = best.get("landing_page_url") or primary.get("landing_page_url") or work.get("doi") or ""

    return {
        "openalex_id": work.get("id", ""),
        "openalex_display_name": clean_text(work.get("display_name", "")),
        "publication_year": str(work.get("publication_year", "")),
        "type": work.get("type", ""),
        "cited_by_count": str(work.get("cited_by_count", "")),
        "is_oa": str(open_access.get("is_oa", "")),
        "oa_status": open_access.get("oa_status", ""),
        "oa_url": open_access.get("oa_url", "") or "",
        "landing_page_url": landing_page_url or "",
        "pdf_url": pdf_url or "",
        "license": best.get("license") or primary.get("license") or "",
        "source_display_name": ((best.get("source") or primary.get("source") or {}).get("display_name", "")),
    }


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def first_pdf_url(locations: list[dict]) -> str:
    for location in locations:
        if location.get("pdf_url"):
            return location["pdf_url"]
    return ""


def download_pdf(url: str, pdf_dir: Path, row: dict[str, str]) -> str:
    filename = safe_filename(f"{row.get('year') or row.get('publication_year')}_{row['title']}") + ".pdf"
    output_path = pdf_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            content_type = response.headers.get("content-type", "")
            data = response.read()
        if b"%PDF" not in data[:2048] and "pdf" not in content_type.lower():
            return ""
        output_path.write_bytes(data)
        return str(output_path)
    except (HTTPError, URLError, TimeoutError):
        return ""


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    fieldnames = [
        "title",
        "authors",
        "year",
        "doi",
        "openalex_id",
        "publication_year",
        "type",
        "cited_by_count",
        "is_oa",
        "oa_status",
        "license",
        "source_display_name",
        "landing_page_url",
        "pdf_url",
        "pdf_file",
        "lookup_error",
        "source_file",
    ]
    with (output_dir / "openalex_enriched_refs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "openalex_enriched_refs.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    lines = [
        "# OpenAlex Reference Export Summary",
        "",
        f"Total records: {len(rows)}",
        f"Open access records: {sum(row.get('is_oa') == 'True' for row in rows)}",
        f"Records with PDF URL: {sum(bool(row.get('pdf_url')) for row in rows)}",
        "",
        "## Records",
        "",
        "| Year | Title | OA | PDF | DOI |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        pdf = "yes" if row.get("pdf_url") else "no"
        oa = row.get("oa_status") or row.get("is_oa") or ""
        title = row["title"].replace("|", "\\|")
        doi = row.get("doi", "")
        doi_link = f"[doi](https://doi.org/{doi})" if doi else ""
        lines.append(f"| {row.get('year', '')} | {title} | {oa} | {pdf} | {doi_link} |")
    (output_dir / "openalex_enriched_refs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_doi(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    return value.strip()


def clean_text(value: str | None) -> str:
    return " ".join((value or "").replace("\xa0", " ").split())


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    return value[:160] or "paper"


if __name__ == "__main__":
    main()
