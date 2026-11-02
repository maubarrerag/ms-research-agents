"""Europe PMC search (free, no API key). Covers PubMed plus preprints."""
from __future__ import annotations

import httpx

URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search(query: str, limit: int = 10) -> list[dict]:
    params = {"query": query, "format": "json", "pageSize": min(limit, 25), "resultType": "core"}
    r = httpx.get(URL, params=params, timeout=30)
    r.raise_for_status()
    out = []
    for item in r.json().get("resultList", {}).get("result", []):
        abstract = (item.get("abstractText") or "").replace("<h4>", "").replace("</h4>", " ")
        out.append({
            "pmid": item.get("pmid"),
            "doi": item.get("doi"),
            "title": item.get("title"),
            "journal": item.get("journalTitle"),
            "year": item.get("pubYear"),
            "cited_by": item.get("citedByCount"),
            "abstract": abstract[:1500],
        })
    return out
