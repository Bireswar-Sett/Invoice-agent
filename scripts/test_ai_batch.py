#!/usr/bin/env python3
"""Exercise the real AI decision call on 12 synthetic invoices; never contacts the grader."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow execution as `python scripts/test_ai_batch.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from invoice_agent import create_app
from invoice_agent.ai import call_openai_compatible, validate_decision


def make_package(index: int, action: str) -> dict:
    vendor = f"Vendor {index:02d} Ltd"
    inv = f"INV-2026-{index:03d}"
    base = [
        f"[INV-{index:03d}] Tax invoice issued by {vendor}; invoice number {inv}; amount INR 12,500.00.",
        f"[PO-{index:03d}] Approved purchase order for {vendor} confirms INR 12,500.00 and the same goods/services.",
        f"[LEDGER-{index:03d}] Accounts-payable ledger confirms {inv} has not previously been paid.",
    ]
    if action == "request_approval":
        base = [
            f"[INV-{index:03d}] Tax invoice issued by {vendor}; invoice number {inv}; amount INR 95,000.00.",
            f"[PO-{index:03d}] Approved purchase order confirms INR 95,000.00 and matching goods/services.",
            f"[POL-{index:03d}] Policy revision TEST-1 delegates autonomous settlement only up to INR 50,000.00; larger valid invoices require approval.",
        ]
    elif action == "hold_invoice":
        base = [
            f"[INV-{index:03d}] Tax invoice issued by {vendor}; invoice number {inv}; amount INR 12,500.00.",
            f"[PO-{index:03d}] Approved purchase order confirms INR 12,500.00 and matching goods/services.",
            f"[VERIFY-{index:03d}] Vendor bank-account change is awaiting independent verification; payment must remain paused until verification completes.",
        ]
    elif action == "reject_duplicate":
        base = [
            f"[INV-{index:03d}] Tax invoice issued by {vendor}; invoice number {inv}; amount INR 12,500.00.",
            f"[PO-{index:03d}] Approved purchase order confirms INR 12,500.00 and matching goods/services.",
            f"[PAY-{index:03d}] Bank payment confirmation shows this exact commercial invoice number {inv} was already paid in full.",
        ]
    elif action == "open_exception":
        base = [
            f"[INV-{index:03d}] Tax invoice issued by {vendor}; invoice number {inv}; amount INR 12,500.00.",
            f"[PO-{index:03d}] Approved purchase order for the same invoice states INR 10,500.00, materially conflicting with the invoice amount.",
            f"[RECON-{index:03d}] Reconciliation report confirms unresolved INR 2,000.00 discrepancy; no corrected purchase order exists.",
        ]
    return {"packageId": f"smoke-pkg-{index:02d}", "documents": base}


def main() -> int:
    actions = [
        "settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception",
        "settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception",
        "settle_invoice", "settle_invoice",
    ]
    packages = [make_package(i + 1, action) for i, action in enumerate(actions)]
    app = create_app({"TESTING": True})
    with app.app_context():
        proposals = call_openai_compatible(app, packages, "TEST-1; autonomous authority INR 50,000.00")
        if set(proposals) != {p["packageId"] for p in packages}:
            raise SystemExit(f"FAIL: expected 12 proposals, got {len(proposals)}")
        failures = []
        for package in packages:
            pid = package["packageId"]
            try:
                validated = validate_decision(package, proposals[pid])
                expected = actions[int(pid.rsplit("-", 1)[1]) - 1]
                if validated["action"] != expected:
                    failures.append(f"{pid}: expected {expected}, got {validated['action']}")
            except Exception as exc:
                failures.append(f"{pid}: {type(exc).__name__}: {exc}")
        print(f"AI batch returned {len(proposals)}/12 proposals")
        if failures:
            print("FAIL: validation/action mismatches:")
            for failure in failures:
                print(" -", failure)
            return 1
        print("PASS: all 12 synthetic proposals passed schema/evidence validation and expected-action checks")
        print("No task was created and the grader was not contacted.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
