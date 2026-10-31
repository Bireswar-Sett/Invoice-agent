from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def principal_from_bearer(token: str) -> str:
    # Never store the bearer token itself. This stable one-way digest is the tenant key.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def extract_batch_part(message: dict) -> dict:
    parts = message.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Message parts must be an array.")
    matching = [
        part for part in parts
        if isinstance(part, dict)
        and part.get("mediaType") == "application/vnd.ga5.invoice-claim-batch+json"
        and isinstance(part.get("data"), dict)
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one invoice batch data part.")
    return matching[0]["data"]


def extract_results_part(message: dict) -> dict:
    parts = message.get("parts")
    if not isinstance(parts, list):
        raise ValueError("Message parts must be an array.")
    matching = [
        part for part in parts
        if isinstance(part, dict)
        and part.get("mediaType") == "application/vnd.ga5.invoice-action-results+json"
        and isinstance(part.get("data"), dict)
    ]
    if len(matching) != 1:
        raise ValueError("Expected exactly one invoice results data part.")
    return matching[0]["data"]


def package_id(package: dict) -> str:
    for key in ("packageId", "package_id", "id"):
        value = package.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Every package must contain a non-empty packageId.")


def collect_bracketed_refs(value: Any) -> set[str]:
    """Find exact bracketed reference tokens in arbitrary nested package data."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            found.update(re.findall(r"\[[^\[\]\r\n]{1,160}\]", node))
        elif isinstance(node, dict):
            for key, item in node.items():
                walk(key)
                walk(item)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)
    return found


def normalize_reference(reference: Any, available: set[str]) -> str:
    if not isinstance(reference, str):
        raise ValueError("Evidence references must be strings.")
    ref = reference.strip()
    if ref in available:
        return ref
    # Some models omit the literal square brackets even though the document uses them.
    candidate = f"[{ref}]"
    if candidate in available:
        return candidate
    raise ValueError("An evidence reference is not present in the package.")


def task_to_dict(task) -> dict:
    import copy

    return {
        "id": task.id,
        "contextId": task.context_id,
        "status": {
            "state": task.state,
            "timestamp": task.status_timestamp,
        },
        "history": copy.deepcopy(task.history or []),
        "artifacts": copy.deepcopy(task.artifacts or []),
    }


def error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}
