from __future__ import annotations

import json
import logging

import pytest
import requests

from invoice_agent.ai import DecisionError, call_openai_compatible


class FakeApp:
    def __init__(self, base_url="https://aipipe.org/openai/v1"):
        self.config = {
            "AI_API_KEY": "test-token-do-not-log",
            "AI_BASE_URL": base_url,
            "AI_MODEL": "gpt-5-nano",
            "AI_TIMEOUT_SECONDS": 10,
            "AI_MAX_OUTPUT_TOKENS": 12000,
            "AI_SYSTEM_PROMPT": "Return JSON only.",
        }
        self.logger = logging.getLogger("test-ai-provider")


def _response(status: int, body: dict) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode("utf-8")
    response.headers["Content-Type"] = "application/json"
    response.url = "https://aipipe.org/openai/v1/chat/completions"
    return response


def test_ai_pipe_openai_chat_completion_endpoint_and_json_mode(monkeypatch):
    captured = {}
    body = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "proposals": [{"packageId": "pkg-1", "action": "settle_invoice"}]
                })
            }
        }]
    }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _response(200, body)

    monkeypatch.setattr("invoice_agent.ai.requests.post", fake_post)
    result = call_openai_compatible(
        FakeApp(),
        [{"packageId": "pkg-1", "documents": ["[INV-1] sample"]}],
        "policy-v1",
    )

    assert captured["url"] == "https://aipipe.org/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-token-do-not-log"
    assert captured["json"]["model"] == "gpt-5-nano"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["max_completion_tokens"] == 12000
    assert "temperature" not in captured["json"]
    assert "max_tokens" not in captured["json"]
    assert set(result) == {"pkg-1"}


def test_ai_provider_http_errors_are_logged_without_token(monkeypatch, caplog):
    response = _response(401, {"error": {"message": "Invalid AI Pipe token"}})

    def fake_post(_url, **_kwargs):
        return response

    monkeypatch.setattr("invoice_agent.ai.requests.post", fake_post)
    caplog.set_level(logging.ERROR, logger="test-ai-provider")

    with pytest.raises(DecisionError, match="HTTP 401"):
        call_openai_compatible(
            FakeApp(),
            [{"packageId": "pkg-1", "documents": ["[INV-1] sample"]}],
            "policy-v1",
        )

    assert "status=401" in caplog.text
    assert "test-token-do-not-log" not in caplog.text
    assert "Invalid AI Pipe token" in caplog.text
