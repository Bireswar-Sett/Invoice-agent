from __future__ import annotations

from flask import current_app, request
from urllib.parse import urlsplit
from sqlalchemy.exc import IntegrityError

from invoice_agent import db
from invoice_agent.models import IdempotencyRecord, TaskRecord
from invoice_agent.service import (
    RequestConflict,
    RequestInvalid,
    cancel_task,
    create_initial_task,
    process_continuation,
)
from invoice_agent.utils import (
    error_payload,
    extract_batch_part,
    extract_results_part,
    principal_from_bearer,
    sha256_json,
    task_to_dict,
)
from invoice_agent import _json_response


def _auth_and_version(require_json_content_type: bool = False):
    authorization = request.headers.get("Authorization", "")
    pieces = authorization.strip().split()
    if len(pieces) != 2 or pieces[0].lower() != "bearer" or not pieces[1]:
        return None, (_json_response(error_payload("UNAUTHENTICATED", "A Bearer token is required."), 401), 401)
    token = pieces[1]
    # Require a single exact bearer value; whitespace-separated extra data is rejected above.
    principal = principal_from_bearer(token)

    version = request.headers.get("A2A-Version")
    if version != "1.0":
        return None, (_json_response(error_payload("A2A_VERSION", "A2A-Version must be 1.0."), 400), 400)

    if require_json_content_type:
        content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/a2a+json":
            return None, (_json_response(error_payload("UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/a2a+json."), 415), 415)
    return principal, None


def _validate_message_envelope(message):
    if not isinstance(message, dict):
        raise RequestInvalid("Request must contain a message object.")
    if not isinstance(message.get("messageId"), str) or not message["messageId"].strip():
        raise RequestInvalid("messageId is required.")
    if len(message["messageId"]) > 256:
        raise RequestInvalid("messageId is too long.")
    if message.get("role") != "ROLE_USER":
        raise RequestInvalid("Only ROLE_USER messages are accepted on this endpoint.")
    if not isinstance(message.get("parts"), list) or not message["parts"]:
        raise RequestInvalid("Message parts must be a non-empty array.")


def register_routes(app):
    base = app.config["A2A_BASE_PATH"]

    @app.get("/.well-known/agent-card.json")
    def agent_card():
        public_base = current_app.config.get("PUBLIC_BASE_URL") or __import__("os").getenv("PUBLIC_BASE_URL")
        if not public_base:
            render_url = __import__("os").getenv("RENDER_EXTERNAL_URL", "https://localhost")
            public_base = render_url.rstrip("/") + base + "/"
        public_base = public_base.rstrip("/") + "/"
        parsed_base = urlsplit(public_base)
        if (
            parsed_base.scheme != "https"
            or not parsed_base.netloc
            or parsed_base.username is not None
            or parsed_base.password is not None
            or parsed_base.query
            or parsed_base.fragment
            or parsed_base.path.rstrip("/") != base
        ):
            current_app.logger.error("PUBLIC_BASE_URL must be a public HTTPS URL matching A2A_BASE_PATH")
            from flask import Response
            return Response(
                '{"error":{"code":"INVALID_PUBLIC_BASE_URL","message":"Public base URL configuration is invalid."}}',
                status=500,
                content_type="application/json; charset=utf-8",
            )
        card = {
            "name": "A2A Invoice Action Agent",
            "description": "An evidence-based invoice decision agent that proposes actions and executes only accepted proposals.",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            "supportedInterfaces": [{
                "url": public_base,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }],
            "defaultInputModes": [
                "application/vnd.ga5.invoice-claim-batch+json",
                "application/vnd.ga5.invoice-action-results+json",
            ],
            "defaultOutputModes": [
                "application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json",
            ],
            "skills": [{
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Reads invoice packages, proposes evidence-backed actions, and records accepted results safely.",
                "tags": ["invoice", "accounts-payable", "evidence", "approval", "idempotency"],
            }],
        }
        from flask import Response
        import json
        return Response(json.dumps(card, separators=(",", ":")), content_type="application/json; charset=utf-8")

    @app.post(base + "/message:send")
    def message_send():
        principal, error = _auth_and_version(require_json_content_type=True)
        if error:
            return error[0]
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _json_response(error_payload("INVALID_ARGUMENT", "Request body must be a JSON object."), 400)
        message = body.get("message")
        try:
            _validate_message_envelope(message)
            # A single preflight handles initial and continuation replays before any parsing,
            # AI call, receipt write, or terminal-state check. The hash intentionally covers
            # message only; request configuration is not part of idempotency.
            message_hash = sha256_json(message)
            existing = IdempotencyRecord.query.filter_by(
                principal=principal,
                message_id=message["messageId"],
            ).first()
            if existing:
                if existing.message_hash != message_hash:
                    return _json_response(error_payload("IDEMPOTENCY_CONFLICT", "This messageId was already used with different message content."), 409)
                existing_task = TaskRecord.query.filter_by(
                    id=existing.task_id,
                    principal=principal,
                ).first()
                if existing_task is not None:
                    return _json_response({"task": task_to_dict(existing_task)}, 200)
                return _json_response(error_payload("PROCESSING_STATE", "The stored request state is unavailable."), 500)

            if message.get("taskId"):
                results_data = extract_results_part(message)
                task, replay = process_continuation(principal, message, results_data)
            else:
                batch_data = extract_batch_part(message)
                task, replay = create_initial_task(current_app, principal, message, batch_data)
            return _json_response({"task": task_to_dict(task)}, 200)
        except RequestConflict as exc:
            code = str(exc) if str(exc) in {"IDEMPOTENCY_CONFLICT", "TASK_NOT_AWAITING_INPUT", "TASK_ALREADY_TERMINAL"} else "CONFLICT"
            db.session.rollback()
            return _json_response(error_payload(code, "The request conflicts with the stored task state."), 409)
        except LookupError:
            db.session.rollback()
            return _json_response(error_payload("NOT_FOUND", "Task not found."), 404)
        except RequestInvalid as exc:
            db.session.rollback()
            return _json_response(error_payload("INVALID_ARGUMENT", str(exc)), 400)
        except ValueError as exc:
            db.session.rollback()
            return _json_response(error_payload("INVALID_ARGUMENT", str(exc)), 400)
        except IntegrityError:
            db.session.rollback()
            # Handles concurrent equivalent initial messages. The unique idempotency key
            # makes one worker the creator; all others replay its complete Task.
            message_id = message.get("messageId") if isinstance(message, dict) else None
            if message_id:
                existing = IdempotencyRecord.query.filter_by(principal=principal, message_id=message_id).first()
                if existing:
                    if existing.message_hash != sha256_json(message):
                        return _json_response(error_payload("IDEMPOTENCY_CONFLICT", "This messageId was already used with different message content."), 409)
                    task = TaskRecord.query.filter_by(id=existing.task_id, principal=principal).first()
                    if task:
                        return _json_response({"task": task_to_dict(task)}, 200)
            return _json_response(error_payload("CONFLICT", "A concurrent request conflicted with stored state."), 409)
        except Exception as exc:
            db.session.rollback()
            current_app.logger.exception("Unhandled message processing failure: %s", exc.__class__.__name__)
            return _json_response(error_payload("PROCESSING_FAILED", "The message could not be processed."), 502)

    @app.get(base + "/tasks/<task_id>")
    def get_task(task_id):
        principal, error = _auth_and_version()
        if error:
            return error[0]
        task = TaskRecord.query.filter_by(id=task_id, principal=principal).first()
        if task is None:
            return _json_response(error_payload("NOT_FOUND", "Task not found."), 404)
        return _json_response(task_to_dict(task), 200)

    @app.get(base + "/tasks")
    def list_tasks():
        principal, error = _auth_and_version()
        if error:
            return error[0]
        tasks = TaskRecord.query.filter_by(principal=principal).order_by(TaskRecord.created_at.desc()).all()
        return _json_response({"tasks": [task_to_dict(task) for task in tasks]}, 200)

    @app.post(base + "/tasks/<task_id>:cancel")
    def cancel(task_id):
        principal, error = _auth_and_version(require_json_content_type=True)
        if error:
            return error[0]
        try:
            task = cancel_task(principal, task_id)
            return _json_response(task_to_dict(task), 200)
        except LookupError:
            db.session.rollback()
            return _json_response(error_payload("NOT_FOUND", "Task not found."), 404)
        except RequestConflict:
            db.session.rollback()
            return _json_response(error_payload("TASK_ALREADY_TERMINAL", "The task is already terminal."), 409)
        except Exception as exc:
            db.session.rollback()
            current_app.logger.exception("Task cancellation failed: %s", exc.__class__.__name__)
            return _json_response(error_payload("CANCEL_FAILED", "The task could not be canceled."), 500)
