from __future__ import annotations

import uuid
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from invoice_agent import db
from invoice_agent.ai import DecisionError, call_openai_compatible, validate_decision
from invoice_agent.models import (
    DecisionCache,
    ExecutionRecord,
    IdempotencyRecord,
    TaskRecord,
    utcnow,
)
from invoice_agent.utils import iso_now, package_id, sha256_json

INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
TERMINAL_STATES = {COMPLETED, CANCELED}
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"


class RequestConflict(Exception):
    pass


class RequestInvalid(Exception):
    pass


def package_content_hash(package: dict, policy_revision) -> str:
    # Package/task IDs are routing namespaces, not invoice evidence. Excluding the
    # top-level package ID lets the same document content reuse its semantic decision
    # when the grader re-namespaces the package. The policy revision remains in the key.
    content = {k: v for k, v in package.items() if k not in {"packageId", "package_id", "id"}}
    return sha256_json({"policyRevision": policy_revision, "package": content})


def _cached_decision(package_hash: str):
    row = db.session.get(DecisionCache, package_hash)
    return row.decision if row else None


def _store_cached_decision(package_hash: str, decision: dict) -> None:
    if db.session.get(DecisionCache, package_hash):
        return
    # A savepoint keeps a simultaneous cache insert from aborting the task transaction.
    try:
        with db.session.begin_nested():
            db.session.add(DecisionCache(package_hash=package_hash, decision=decision))
            db.session.flush()
    except IntegrityError:
        # Another worker won the unique-key race. Its canonical decision is reusable.
        return


def build_proposals(app, packages: list[dict], policy_revision) -> list[dict]:
    package_hashes = [package_content_hash(package, policy_revision) for package in packages]
    decisions: dict[str, dict] = {}
    misses: list[dict] = []

    for package, p_hash in zip(packages, package_hashes):
        cached = _cached_decision(p_hash)
        if cached is None:
            misses.append(package)
        else:
            # The cache stores semantic fields without package/task IDs. Bind it back to
            # this package, then re-validate evidence and facts against the current content.
            pid = package_id(package)
            decisions[pid] = validate_decision(package, {**cached, "packageId": pid})

    if misses:
        generated = app.config.get("AI_DECIDER")
        if generated:
            raw_by_id = generated(misses, policy_revision)
        else:
            raw_by_id = call_openai_compatible(app, misses, policy_revision)
        expected_ids = {package_id(p) for p in misses}
        if set(raw_by_id) != expected_ids:
            raise DecisionError("AI output package IDs do not exactly match the uncached input packages.")

        for package in misses:
            pid = package_id(package)
            decision = validate_decision(package, raw_by_id[pid])
            decisions[pid] = decision
            p_hash = package_content_hash(package, policy_revision)
            _store_cached_decision(p_hash, decision)

    proposals = []
    used_ids = set()
    for package in packages:
        pid = package_id(package)
        decision = decisions[pid]
        action_id = "act_" + uuid.uuid4().hex
        while action_id in used_ids:
            action_id = "act_" + uuid.uuid4().hex
        used_ids.add(action_id)
        proposals.append({
            "packageId": pid,
            "actionId": action_id,
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"],
        })
    return proposals


def _proposal_artifact(batch_id: str, proposals: list[dict]) -> dict:
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "invoice-action-proposals",
        "parts": [{
            "mediaType": PROPOSAL_MEDIA,
            "data": {"batchId": batch_id, "proposals": proposals},
        }],
    }


def _receipt_artifact(batch_id: str, executions: list[dict]) -> dict:
    return {
        "artifactId": str(uuid.uuid4()),
        "name": "invoice-action-receipts",
        "parts": [{
            "mediaType": RECEIPT_MEDIA,
            "data": {"batchId": batch_id, "executions": executions},
        }],
    }


def find_idempotency(principal: str, message_id: str):
    return IdempotencyRecord.query.filter_by(
        principal=principal,
        message_id=message_id,
    ).first()


def replay_or_conflict(principal: str, message_id: str, message_hash: str):
    record = find_idempotency(principal, message_id)
    if not record:
        return None
    if record.message_hash != message_hash:
        raise RequestConflict("IDEMPOTENCY_CONFLICT")
    task = TaskRecord.query.filter_by(id=record.task_id, principal=principal).first()
    if task is None:
        # A broken foreign reference is a server integrity problem, not a reason to leak data.
        raise RuntimeError("Idempotency record references a missing task")
    return task


def create_initial_task(app, principal: str, message: dict, batch_data: dict):
    message_id = message["messageId"]
    message_hash = sha256_json(message)
    existing = replay_or_conflict(principal, message_id, message_hash)
    if existing is not None:
        return existing, True

    batch_id = batch_data.get("batchId")
    policy_revision = batch_data.get("policyRevision")
    packages = batch_data.get("packages")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise RequestInvalid("batchId must be a non-empty string.")
    if len(batch_id) > 256:
        raise RequestInvalid("batchId is too long.")
    if policy_revision is None:
        raise RequestInvalid("policyRevision is required.")
    if not isinstance(packages, list) or not packages:
        raise RequestInvalid("packages must be a non-empty array.")
    if len(packages) > app.config["MAX_PACKAGES_PER_BATCH"]:
        raise RequestInvalid("The batch contains too many packages.")
    if any(not isinstance(p, dict) for p in packages):
        raise RequestInvalid("Every package must be a JSON object.")

    ids = [package_id(p) for p in packages]
    if any(len(pid) > 256 for pid in ids):
        raise RequestInvalid("A packageId is too long.")
    if len(ids) != len(set(ids)):
        raise RequestInvalid("Package IDs must be unique within the batch.")

    task_id = str(uuid.uuid4())
    context_id = message.get("contextId")
    if context_id is None:
        context_id = str(uuid.uuid4())
    elif not isinstance(context_id, str) or not context_id.strip():
        raise RequestInvalid("contextId must be a non-empty string when supplied.")
    if len(context_id) > 128:
        raise RequestInvalid("contextId is too long.")
    now = iso_now()
    task = TaskRecord(
        id=task_id,
        principal=principal,
        context_id=context_id,
        batch_id=batch_id,
        state="TASK_STATE_WORKING",
        status_timestamp=now,
        history=[message],
        artifacts=[],
        proposals=[],
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db.session.add(task)
    db.session.add(IdempotencyRecord(
        principal=principal,
        message_id=message_id,
        message_hash=message_hash,
        task_id=task_id,
    ))
    # Force the unique idempotency reservation before expensive model work. A concurrent
    # duplicate request waits on the unique index and then replays the committed Task.
    db.session.flush()

    proposals = build_proposals(app, packages, policy_revision)
    task.proposals = proposals
    task.artifacts = [_proposal_artifact(batch_id, proposals)]
    task.state = INPUT_REQUIRED
    task.status_timestamp = iso_now()
    task.updated_at = utcnow()
    db.session.commit()
    return task, False


def _validate_continuation(task: TaskRecord, message: dict, results_data: dict) -> tuple[list[dict], list[dict]]:
    if message.get("taskId") != task.id:
        raise RequestInvalid("taskId does not match the stored task.")
    if message.get("contextId") != task.context_id:
        raise RequestInvalid("contextId does not match the stored task.")
    if results_data.get("batchId") != task.batch_id:
        raise RequestInvalid("batchId does not match the stored task.")

    results = results_data.get("results")
    proposals = task.proposals or []
    if not isinstance(results, list) or len(results) != len(proposals):
        raise RequestInvalid("Results must contain exactly one result for every proposal.")

    proposal_by_package = {p["packageId"]: p for p in proposals}
    seen_packages: set[str] = set()
    executions: list[dict] = []
    result_records: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            raise RequestInvalid("Every result must be an object.")
        pid = result.get("packageId")
        if not isinstance(pid, str) or pid not in proposal_by_package or pid in seen_packages:
            raise RequestInvalid("A result package does not match exactly one stored proposal.")
        seen_packages.add(pid)
        proposal = proposal_by_package[pid]
        if result.get("actionId") != proposal["actionId"]:
            raise RequestInvalid("A result actionId does not match the stored proposal.")
        if result.get("action") != proposal["action"]:
            raise RequestInvalid("A result action does not match the stored proposal.")
        outcome = result.get("outcome")
        nonce = result.get("receiptNonce")
        if outcome not in {"ACCEPTED", "REJECTED"}:
            raise RequestInvalid("Result outcome must be ACCEPTED or REJECTED.")
        if not isinstance(nonce, str) or not nonce.strip() or len(nonce) > 512:
            raise RequestInvalid("Every result must contain a valid receiptNonce.")

        result_records.append({
            "packageId": pid,
            "actionId": proposal["actionId"],
            "action": proposal["action"],
            "outcome": outcome,
            "receiptNonce": nonce,
        })
        if outcome == "ACCEPTED":
            executions.append({
                "packageId": pid,
                "actionId": proposal["actionId"],
                "action": proposal["action"],
                "receiptNonce": nonce,
                "facts": proposal["facts"],
                "evidenceRefs": proposal["evidenceRefs"],
            })

    if seen_packages != set(proposal_by_package):
        raise RequestInvalid("Results do not cover the complete proposal set.")
    return result_records, executions


def process_continuation(principal: str, message: dict, results_data: dict):
    task_id = message.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        raise RequestInvalid("A continuation must include taskId.")

    # The principal predicate prevents both reads and writes across user boundaries.
    task = TaskRecord.query.filter_by(id=task_id, principal=principal).with_for_update().first()
    if task is None:
        raise LookupError("Task not found")

    message_id = message["messageId"]
    message_hash = sha256_json(message)
    existing = replay_or_conflict(principal, message_id, message_hash)
    if existing is not None:
        return existing, True

    if task.state != INPUT_REQUIRED:
        raise RequestConflict("TASK_NOT_AWAITING_INPUT")

    result_records, executions = _validate_continuation(task, message, results_data)
    receipt_artifact = _receipt_artifact(task.batch_id, executions)
    new_history = list(task.history or []) + [message]
    new_artifacts = list(task.artifacts or []) + [receipt_artifact]

    db.session.add(IdempotencyRecord(
        principal=principal,
        message_id=message_id,
        message_hash=message_hash,
        task_id=task.id,
    ))
    db.session.flush()

    # Persist accepted executions inside the same transaction as the atomic state change.
    for execution in executions:
        db.session.add(ExecutionRecord(
            task_id=task.id,
            principal=principal,
            package_id=execution["packageId"],
            action_id=execution["actionId"],
            action=execution["action"],
            receipt_nonce=execution["receiptNonce"],
            execution_data=execution,
        ))

    now = iso_now()
    result = db.session.execute(
        update(TaskRecord)
        .where(
            TaskRecord.id == task.id,
            TaskRecord.principal == principal,
            TaskRecord.state == INPUT_REQUIRED,
        )
        .values(
            state=COMPLETED,
            status_timestamp=now,
            history=new_history,
            artifacts=new_artifacts,
            updated_at=utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        db.session.rollback()
        raise RequestConflict("TASK_ALREADY_TERMINAL")

    db.session.commit()
    # synchronize_session=False avoids loading the large JSON history during the update;
    # refresh the already-loaded identity so the returned Task shows the terminal state.
    db.session.refresh(task)
    return task, False


def cancel_task(principal: str, task_id: str):
    task = TaskRecord.query.filter_by(id=task_id, principal=principal).first()
    if task is None:
        raise LookupError("Task not found")

    now = iso_now()
    result = db.session.execute(
        update(TaskRecord)
        .where(
            TaskRecord.id == task_id,
            TaskRecord.principal == principal,
            TaskRecord.state.notin_(tuple(TERMINAL_STATES)),
        )
        .values(
            state=CANCELED,
            status_timestamp=now,
            updated_at=utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        db.session.rollback()
        raise RequestConflict("TASK_ALREADY_TERMINAL")
    db.session.commit()
    db.session.refresh(task)
    return task
