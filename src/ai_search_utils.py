"""
ai_search_utils.py
==================

Reusable helpers for working with an Azure AI Search (a.k.a. Azure Cognitive
Search) service: clients, index introspection, exhaustive facet / distinct-value
extraction (past the 1000-value cap), full-index document iteration (past the
100k skip cap), and batched write operations.

Install
-------
    pip install azure-search-documents

All functions take an explicit client so they stay stateless and testable.
Nothing here holds credentials; build clients with the factories below or your
own (e.g. DefaultAzureCredential for AAD/Managed Identity).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

Number = Union[int, float]
JSON = Dict[str, Any]


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
def get_search_client(endpoint: str, index_name: str, api_key: str) -> SearchClient:
    """Build a SearchClient for a single index using an admin or query key."""
    return SearchClient(endpoint, index_name, AzureKeyCredential(api_key))


def get_index_client(endpoint: str, api_key: str) -> SearchIndexClient:
    """Build a SearchIndexClient (service/index-level ops) using an admin key."""
    return SearchIndexClient(endpoint, AzureKeyCredential(api_key))


# ---------------------------------------------------------------------------
# OData helpers
# ---------------------------------------------------------------------------
def odata_literal(value: Any, is_string: bool = True) -> str:
    """Render a Python value as an OData literal (escapes single quotes)."""
    if value is None:
        return "null"
    if is_string:
        return "'" + str(value).replace("'", "''") + "'"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def and_filters(*clauses: Optional[str]) -> Optional[str]:
    """Combine non-empty filter clauses with AND, wrapping each in parens."""
    parts = [f"({c})" for c in clauses if c]
    return " and ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Index introspection
# ---------------------------------------------------------------------------
def count_documents(search_client: SearchClient, filter: Optional[str] = None) -> int:
    """Total document count (optionally within an OData filter)."""
    result = search_client.search(
        search_text="*", filter=filter, top=0, include_total_count=True
    )
    return result.get_count()


def get_index_stats(index_client: SearchIndexClient, index_name: str) -> JSON:
    """Return {'document_count', 'storage_size_bytes', ...} for an index."""
    stats = index_client.get_index_statistics(index_name)
    # SDK returns a dict-like; normalise the two headline numbers.
    return {
        "document_count": stats.get("document_count", stats.get("documentCount")),
        "storage_size_bytes": stats.get(
            "storage_size", stats.get("storageSize")
        ),
        "raw": stats,
    }


def get_field_map(index_client: SearchIndexClient, index_name: str) -> Dict[str, JSON]:
    """
    Return {field_name: {type, key, filterable, facetable, sortable,
    searchable, retrievable, collection}} for every top-level field.
    """
    index = index_client.get_index(index_name)
    out: Dict[str, JSON] = {}
    for f in index.fields:
        ftype = getattr(f, "type", "")
        out[f.name] = {
            "type": ftype,
            "collection": str(ftype).startswith("Collection("),
            "key": bool(getattr(f, "key", False)),
            "filterable": bool(getattr(f, "filterable", False)),
            "facetable": bool(getattr(f, "facetable", False)),
            "sortable": bool(getattr(f, "sortable", False)),
            "searchable": bool(getattr(f, "searchable", False)),
            "retrievable": not bool(getattr(f, "hidden", False)),
        }
    return out


def get_key_field(index_client: SearchIndexClient, index_name: str) -> str:
    """Return the name of the index's key field."""
    index = index_client.get_index(index_name)
    for f in index.fields:
        if getattr(f, "key", False):
            return f.name
    raise ValueError(f"No key field found on index '{index_name}'.")


def list_facetable_fields(index_client: SearchIndexClient, index_name: str) -> List[str]:
    """Names of all facetable fields (useful before calling facet helpers)."""
    return [n for n, m in get_field_map(index_client, index_name).items() if m["facetable"]]


# ---------------------------------------------------------------------------
# Exhaustive facet / distinct-value extraction (past the 1000 cap)
# ---------------------------------------------------------------------------
def get_all_facet_values(
    search_client: SearchClient,
    field: str,
    is_string: bool = True,
    page_size: int = 1000,
    base_filter: Optional[str] = None,
    search_text: str = "*",
) -> Dict[Any, int]:
    """
    {value: document_count} for every distinct value of a SCALAR field.

    Bypasses the 1000-value facet cap by requesting the facet sorted by value
    and paging with `field gt <last_value>`. Counts stay exact because each
    cursor step only removes values already counted.

    Set is_string=False for numeric fields (so the cursor isn't quoted).
    """
    page_size = min(page_size, 1000)
    results: Dict[Any, int] = {}
    cursor: Any = None

    while True:
        cursor_clause = (
            f"{field} gt {odata_literal(cursor, is_string)}" if cursor is not None else None
        )
        flt = and_filters(base_filter, cursor_clause)

        response = search_client.search(
            search_text=search_text,
            facets=[f"{field},sort:value,count:{page_size}"],
            filter=flt,
            top=0,
            include_total_count=False,
        )
        buckets = (response.get_facets() or {}).get(field, [])
        if not buckets:
            break
        for b in buckets:
            results[b["value"]] = b["count"]
        if len(buckets) < page_size:
            break
        cursor = buckets[-1]["value"]

    return results


def get_all_facet_values_collection(
    search_client: SearchClient,
    field: str,
    page_size: int = 1000,
    base_filter: Optional[str] = None,
    search_text: str = "*",
) -> Dict[str, int]:
    """
    {value: document_count} for a Collection(Edm.String) (multi-valued) field.

    Pages with `field/any(v: v gt cursor)` and accepts only buckets strictly
    beyond the cursor, so boundary values that reappear (docs holding several
    values) aren't double counted.
    """
    page_size = min(page_size, 1000)
    results: Dict[str, int] = {}
    cursor: Optional[str] = None

    while True:
        cursor_clause = (
            f"{field}/any(v: v gt {odata_literal(cursor, True)})" if cursor is not None else None
        )
        flt = and_filters(base_filter, cursor_clause)

        response = search_client.search(
            search_text=search_text,
            facets=[f"{field},sort:value,count:{page_size}"],
            filter=flt,
            top=0,
            include_total_count=False,
        )
        buckets = (response.get_facets() or {}).get(field, [])
        new = [b for b in buckets if cursor is None or b["value"] > cursor]
        if not new:
            break
        for b in new:
            results[b["value"]] = b["count"]
        if len(buckets) < page_size:
            break
        cursor = max(b["value"] for b in buckets)

    return results


def value_counts(
    search_client: SearchClient,
    field: str,
    is_collection: bool = False,
    is_string: bool = True,
    base_filter: Optional[str] = None,
    top: Optional[int] = None,
) -> List[tuple]:
    """
    pandas-style value_counts: list of (value, count) sorted by count desc.
    Auto-dispatches to the scalar or collection extractor.
    """
    counts = (
        get_all_facet_values_collection(search_client, field, base_filter=base_filter)
        if is_collection
        else get_all_facet_values(search_client, field, is_string=is_string, base_filter=base_filter)
    )
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return ordered[:top] if top else ordered


# ---------------------------------------------------------------------------
# Full-index document iteration (past the 100k skip cap)
# ---------------------------------------------------------------------------
def iterate_all_documents(
    search_client: SearchClient,
    key_field: str,
    select: Optional[Sequence[str]] = None,
    base_filter: Optional[str] = None,
    search_text: str = "*",
    page_size: int = 1000,
    key_is_string: bool = True,
) -> Iterator[JSON]:
    """
    Yield every matching document, paging with a cursor on the key field so the
    100,000-result deep-paging limit never applies.

    Requires `key_field` to be sortable & filterable (the document key usually
    is, or add a sortable/filterable field to sort on). Sorts ascending by key.
    """
    page_size = min(page_size, 1000)
    cursor: Any = None

    while True:
        cursor_clause = (
            f"{key_field} gt {odata_literal(cursor, key_is_string)}" if cursor is not None else None
        )
        flt = and_filters(base_filter, cursor_clause)

        results = search_client.search(
            search_text=search_text,
            filter=flt,
            select=list(select) if select else None,
            order_by=[f"{key_field} asc"],
            top=page_size,
            include_total_count=False,
        )
        rows = list(results)
        if not rows:
            break
        for row in rows:
            yield dict(row)
        if len(rows) < page_size:
            break
        cursor = rows[-1][key_field]


def get_all_key_values(
    search_client: SearchClient,
    key_field: str,
    base_filter: Optional[str] = None,
    key_is_string: bool = True,
) -> List[Any]:
    """Collect all document keys (handy for bulk delete / diffing indexes)."""
    return [
        doc[key_field]
        for doc in iterate_all_documents(
            search_client,
            key_field,
            select=[key_field],
            base_filter=base_filter,
            key_is_string=key_is_string,
        )
    ]


# ---------------------------------------------------------------------------
# Batched write operations
# ---------------------------------------------------------------------------
def _chunks(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _batched_action(
    search_client: SearchClient,
    action: str,               # 'upload' | 'merge' | 'mergeOrUpload' | 'delete'
    documents: Sequence[JSON],
    batch_size: int = 1000,
    max_retries: int = 3,
) -> Dict[str, int]:
    """Run a write action in batches with simple retry; returns success/fail counts."""
    method = {
        "upload": search_client.upload_documents,
        "merge": search_client.merge_documents,
        "mergeOrUpload": search_client.merge_or_upload_documents,
        "delete": search_client.delete_documents,
    }[action]

    succeeded = failed = 0
    for batch in _chunks(list(documents), min(batch_size, 1000)):
        attempt = 0
        while True:
            try:
                results = method(documents=batch)
                succeeded += sum(1 for r in results if r.succeeded)
                failed += sum(1 for r in results if not r.succeeded)
                break
            except Exception:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(2 ** attempt)
    return {"succeeded": succeeded, "failed": failed}


def upload_documents(search_client, documents, batch_size=1000, max_retries=3):
    """Insert/replace documents in batches."""
    return _batched_action(search_client, "upload", documents, batch_size, max_retries)


def merge_or_upload_documents(search_client, documents, batch_size=1000, max_retries=3):
    """Partial update, inserting when the key is new."""
    return _batched_action(search_client, "mergeOrUpload", documents, batch_size, max_retries)


def delete_documents_by_key(
    search_client: SearchClient,
    key_field: str,
    keys: Iterable[Any],
    batch_size: int = 1000,
) -> Dict[str, int]:
    """Delete documents given an iterable of key values."""
    docs = [{key_field: k} for k in keys]
    return _batched_action(search_client, "delete", docs, batch_size)


def delete_documents_by_filter(
    search_client: SearchClient,
    key_field: str,
    filter: str,
    key_is_string: bool = True,
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    Delete every document matching an OData filter (queries keys first, since
    the API has no direct delete-by-query).
    """
    keys = get_all_key_values(
        search_client, key_field, base_filter=filter, key_is_string=key_is_string
    )
    if not keys:
        return {"succeeded": 0, "failed": 0}
    return delete_documents_by_key(search_client, key_field, keys, batch_size)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ENDPOINT = "https://<your-service>.search.windows.net"
    ADMIN_KEY = "<admin-or-query-key>"
    INDEX = "<your-index>"

    sc = get_search_client(ENDPOINT, INDEX, ADMIN_KEY)
    ic = get_index_client(ENDPOINT, ADMIN_KEY)

    print("Docs:", count_documents(sc))
    print("Stats:", get_index_stats(ic, INDEX))
    print("Facetable fields:", list_facetable_fields(ic, INDEX))

    for value, n in value_counts(sc, "<your-facetable-field>", top=20):
        print(f"{n:>10}  {value}")
