#!/usr/bin/env python3
"""Safe diagnostics for the A2A invoice agent and AI Pipe.

This script does not send invoice batches, create tasks, or call the grader.
It only performs public/invalid-request probes against the agent and an optional
small, non-invoice AI Pipe connectivity test.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from urllib.parse import urlsplit

import requests

PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MEDIA = "application/vnd.ga5.invoice-action-results+json"
A2A_MEDIA = "application/a2a+json"
DEFAULT_BASE_URL = "https://a2a-invoice-agent-wr8o.onrender.com"


def show(name: str, ok: bool, detail: str) -> bool:
    print(f"{'PASS' if ok else 'FAIL':4}  {name}: {detail}")
    return ok


def safe_body(response: requests.Response, limit: int = 500) -> str:
    body = response.text[:limit].replace("\n", " ")
    body = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", body)
    body = re.sub(r"(?i)(api[_-]?key|token)(\s*[=:]\s*)[^,\s\"']+", r"\1\2[REDACTED]", body)
    return body


def request_probe(name: str, method: str, url: str, expected: int, **kwargs) -> tuple[bool, requests.Response | None]:
    try:
        response = requests.request(method, url, timeout=(5, 20), allow_redirects=False, **kwargs)
        ok = response.status_code == expected
        detail = f"HTTP {response.status_code} (expected {expected})"
        if not ok:
            detail += f"; body={safe_body(response)}"
        return show(name, ok, detail), response
    except requests.RequestException as exc:
        return show(name, False, f"{type(exc).__name__}: {exc}"), None


def test_agent(base_url: str, token: str) -> bool:
    origin = base_url.rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        print("FAIL  base URL must be an origin only, e.g. https://host.onrender.com")
        return False

    passed = []
    health_ok, health = request_probe("health", "GET", origin + "/healthz", 200)
    passed.append(health_ok)
    if health is not None and health_ok:
        try:
            passed.append(show("health JSON", health.json().get("status") == "ok", str(health.json())))
        except (ValueError, AttributeError):
            passed.append(show("health JSON", False, "response was not JSON"))

    card_ok, card_response = request_probe(
        "public Agent Card", "GET", origin + "/.well-known/agent-card.json", 200
    )
    passed.append(card_ok)
    if card_response is not None and card_ok:
        try:
            card = card_response.json()
            def interface_matches(item):
                candidate = urlsplit(item.get("url", ""))
                exact_remote_origin = candidate.netloc == parsed.netloc and candidate.scheme == parsed.scheme
                local_origin = parsed.hostname in {"localhost", "127.0.0.1"} and candidate.path.rstrip("/") == "/a2a" and candidate.scheme == "https"
                return (
                    candidate.path.rstrip("/") == "/a2a"
                    and (exact_remote_origin or local_origin)
                    and item.get("protocolBinding") == "HTTP+JSON"
                    and item.get("protocolVersion") == "1.0"
                )
            interface = next(item for item in card.get("supportedInterfaces", []) if interface_matches(item))
            card_ok_fields = bool(card.get("name")) and bool(card.get("description")) and bool(card.get("version")) and isinstance(card.get("capabilities"), dict)
            skill_ok = any(
                skill.get("id") == "invoice_action_agent"
                and skill.get("name") and skill.get("description") and skill.get("tags")
                for skill in card.get("skills", [])
            )
            modes_ok = BATCH_MEDIA in card.get("defaultInputModes", []) and PROPOSAL_MEDIA in card.get("defaultOutputModes", []) and RECEIPT_MEDIA in card.get("defaultOutputModes", [])
            passed.append(show("Agent Card contract", bool(interface) and card_ok_fields and skill_ok and modes_ok, "interface, skill, capabilities and media modes"))
        except (ValueError, StopIteration, AttributeError, TypeError):
            passed.append(show("Agent Card contract", False, "card JSON missing required fields or exact /a2a/ interface"))

    message_url = origin + "/a2a/message:send"
    dummy_message = {"message": {"messageId": "diag-no-side-effects", "role": "ROLE_USER", "parts": []}}
    # Each probe intentionally fails before invoice extraction or AI invocation.
    passed.append(request_probe("POST without auth", "POST", message_url, 401, json=dummy_message, headers={"A2A-Version": "1.0", "Content-Type": A2A_MEDIA})[0])
    passed.append(request_probe("POST missing A2A-Version", "POST", message_url, 400, json=dummy_message, headers={"Authorization": f"Bearer {token}", "Content-Type": A2A_MEDIA})[0])
    passed.append(request_probe("POST wrong A2A-Version", "POST", message_url, 400, json=dummy_message, headers={"Authorization": f"Bearer {token}", "A2A-Version": "999", "Content-Type": A2A_MEDIA})[0])
    passed.append(request_probe("POST wrong Content-Type", "POST", message_url, 415, data=json.dumps(dummy_message), headers={"Authorization": f"Bearer {token}", "A2A-Version": "1.0", "Content-Type": "application/json"})[0])

    invalid_message = {
        "message": {
            "messageId": "diag-invalid-envelope",
            "role": "ROLE_USER",
            "parts": [{"mediaType": "text/plain", "text": "diagnostic only; no invoice packages"}],
        },
        "configuration": {"returnImmediately": False},
    }
    passed.append(request_probe("POST invalid part (no AI/no task)", "POST", message_url, 400, json=invalid_message, headers={"Authorization": f"Bearer {token}", "A2A-Version": "1.0", "Content-Type": A2A_MEDIA})[0])

    auth_headers = {"Authorization": f"Bearer {token}", "A2A-Version": "1.0"}
    list_ok, list_response = request_probe("owner task list", "GET", origin + "/a2a/tasks", 200, headers=auth_headers)
    passed.append(list_ok)
    if list_response is not None and list_ok:
        try:
            data = list_response.json()
            passed.append(show("task list envelope", isinstance(data.get("tasks"), list), f"tasks={len(data.get('tasks', []))}"))
        except (ValueError, AttributeError):
            passed.append(show("task list envelope", False, "response was not JSON"))

    not_found = "diagnostic-task-does-not-exist"
    passed.append(request_probe("unknown task read (no mutation)", "GET", origin + f"/a2a/tasks/{not_found}", 404, headers=auth_headers)[0])
    passed.append(request_probe("unknown task cancel (no mutation)", "POST", origin + f"/a2a/tasks/{not_found}:cancel", 404, headers={**auth_headers, "Content-Type": A2A_MEDIA}, json={})[0])

    print(f"\nA2A diagnostic result: {sum(passed)}/{len(passed)} probes passed")
    return all(passed)


def test_ai_pipe() -> bool:
    api_key = os.getenv("AI_API_KEY", "")
    base = os.getenv("AI_BASE_URL", "https://aipipe.org/openrouter/v1").rstrip("/")
    model = os.getenv("AI_MODEL", "openai/gpt-4.1-nano")
    timeout = float(os.getenv("AI_DIAGNOSTIC_TIMEOUT", "20"))
    if not api_key:
        print("FAIL  AI Pipe smoke test: AI_API_KEY is not set in this shell")
        return False
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    print(f"\nAI provider smoke test: host={urlsplit(url).hostname}, model={model}")
    payload = {
        "model": model,
        "temperature": 0,
        "max_completion_tokens": 100,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return this JSON object: {\"ok\": true, \"provider\": \"aipipe\"}."},
        ],
    }
    started = time.monotonic()
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(5, timeout),
        )
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            print(f"FAIL  AI Pipe HTTP {response.status_code} after {elapsed:.2f}s: {safe_body(response)}")
            return False
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        ok = isinstance(parsed, dict) and parsed.get("ok") is True
        print(f"{'PASS' if ok else 'FAIL'}  AI Pipe completion after {elapsed:.2f}s; returned JSON={parsed!r}")
        return ok
    except requests.Timeout:
        print(f"FAIL  AI Pipe request timed out after {timeout}s")
        return False
    except requests.RequestException as exc:
        print(f"FAIL  AI Pipe transport error: {type(exc).__name__}: {exc}")
        return False
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"FAIL  AI Pipe returned unexpected JSON/envelope: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("A2A_AGENT_URL", DEFAULT_BASE_URL), help="Origin only; no /a2a path")
    parser.add_argument("--token", default=os.getenv("A2A_PROBE_TOKEN", "local-diagnostic-probe-token"), help="A dummy opaque token used only for no-side-effect authenticated probes")
    parser.add_argument("--test-ai", action="store_true", help="Run a tiny AI Pipe completion using AI_API_KEY from environment")
    args = parser.parse_args()
    ok = test_agent(args.base_url, args.token)
    if args.test_ai:
        ok = test_ai_pipe() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
