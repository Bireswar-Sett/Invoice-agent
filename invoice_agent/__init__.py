from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

load_dotenv()


db = SQLAlchemy(session_options={"expire_on_commit": False})


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Enable foreign keys and a busy timeout for local SQLite development."""
    try:
        module = dbapi_connection.__class__.__module__
        if "sqlite3" in module:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()
    except Exception:
        logging.getLogger(__name__).exception("Could not configure SQLite connection")


def _database_uri() -> str:
    uri = os.getenv("DATABASE_URL", "sqlite:///.data/invoice-agent.db")
    # Render and several providers still expose postgres:// URLs.
    if uri.startswith("postgres://"):
        uri = "postgresql+psycopg://" + uri[len("postgres://"):]
    elif uri.startswith("postgresql://") and "+" not in uri.split(":", 1)[0]:
        uri = "postgresql+psycopg://" + uri[len("postgresql://"):]
    if uri.startswith("sqlite:///") and ":memory:" not in uri:
        relative = uri[len("sqlite:///"):]
        if not relative.startswith("/"):
            project_root = Path(__file__).resolve().parent.parent
            db_path = (project_root / relative).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            uri = "sqlite:///" + str(db_path)
    return uri


def _normalise_base_path(value: str | None) -> str:
    path = (value or "/a2a").strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.rstrip("/")
    return path or "/a2a"


def _json_response(payload, status=200, content_type="application/a2a+json"):
    # Compact JSON matters: full Task.history can contain the original large batch.
    import json
    from flask import Response

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    max_bytes = int(os.getenv("MAX_RESPONSE_BYTES", "524288"))
    if len(body.encode("utf-8")) > max_bytes:
        body = json.dumps(
            {"error": {"code": "RESPONSE_TOO_LARGE", "message": "Response exceeds the protocol size limit."}},
            separators=(",", ":"),
        )
        status = 500
    return Response(body, status=status, content_type=f"{content_type}; charset=utf-8")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=_database_uri(),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_REQUEST_BYTES", "2097152")),
        A2A_BASE_PATH=_normalise_base_path(os.getenv("A2A_BASE_PATH", "/a2a")),
        AI_BASE_URL=os.getenv("AI_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/"),
        AI_API_KEY=os.getenv("AI_API_KEY", ""),
        AI_MODEL=os.getenv("AI_MODEL", "gpt-5-nano"),
        AI_REASONING_EFFORT=os.getenv("AI_REASONING_EFFORT", "low"),
        AI_TIMEOUT_SECONDS=float(os.getenv("AI_TIMEOUT_SECONDS", "38")),
        AI_MAX_OUTPUT_TOKENS=int(os.getenv("AI_MAX_OUTPUT_TOKENS", "12000")),
        MAX_PACKAGES_PER_BATCH=int(os.getenv("MAX_PACKAGES_PER_BATCH", "60")),
        AI_SYSTEM_PROMPT=os.getenv("AI_SYSTEM_PROMPT", ""),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
    )
    if test_config:
        app.config.update(test_config)

    logging.basicConfig(level=getattr(logging, str(app.config["LOG_LEVEL"]).upper(), logging.INFO))
    from urllib.parse import urlsplit
    _ai_host = urlsplit(app.config["AI_BASE_URL"]).hostname or "unknown"
    app.logger.info(
        "Invoice AI configured provider_host=%s provider_path=%s model=%s reasoning_effort=%s timeout_seconds=%s",
        _ai_host,
        urlsplit(app.config["AI_BASE_URL"]).path,
        app.config["AI_MODEL"],
        app.config.get("AI_REASONING_EFFORT", "low"),
        app.config["AI_TIMEOUT_SECONDS"],
    )
    db.init_app(app)

    from invoice_agent import models  # noqa: F401
    from invoice_agent.routes import register_routes

    with app.app_context():
        db.create_all()

    @app.get("/healthz")
    def healthz():
        # Avoid exposing configuration or secrets through the health endpoint.
        return jsonify({"status": "ok"}), 200

    @app.errorhandler(413)
    def request_too_large(_error):
        return _json_response(
            {"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body exceeds the configured limit."}},
            413,
        )

    register_routes(app)
    return app
