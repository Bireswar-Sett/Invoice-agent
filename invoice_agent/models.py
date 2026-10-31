from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import UniqueConstraint, Index

from invoice_agent import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskRecord(db.Model):
    __tablename__ = "tasks"

    id = db.Column(db.String(64), primary_key=True)
    principal = db.Column(db.String(64), nullable=False, index=True)
    context_id = db.Column(db.String(128), nullable=False)
    batch_id = db.Column(db.String(256), nullable=False)
    state = db.Column(db.String(64), nullable=False, index=True)
    status_timestamp = db.Column(db.String(40), nullable=False)
    history = db.Column(db.JSON, nullable=False, default=list)
    artifacts = db.Column(db.JSON, nullable=False, default=list)
    proposals = db.Column(db.JSON, nullable=False, default=list)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        Index("ix_tasks_principal_created", "principal", "created_at"),
    )


class IdempotencyRecord(db.Model):
    __tablename__ = "idempotency_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    principal = db.Column(db.String(64), nullable=False)
    message_id = db.Column(db.String(256), nullable=False)
    message_hash = db.Column(db.String(64), nullable=False)
    task_id = db.Column(db.String(64), nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("principal", "message_id", name="uq_idempotency_principal_message"),
    )


class DecisionCache(db.Model):
    __tablename__ = "decision_cache"

    package_hash = db.Column(db.String(64), primary_key=True)
    decision = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)


class ExecutionRecord(db.Model):
    __tablename__ = "executions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    task_id = db.Column(db.String(64), nullable=False, index=True)
    principal = db.Column(db.String(64), nullable=False, index=True)
    package_id = db.Column(db.String(256), nullable=False)
    action_id = db.Column(db.String(128), nullable=False, unique=True)
    action = db.Column(db.String(64), nullable=False)
    receipt_nonce = db.Column(db.String(512), nullable=False)
    execution_data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
