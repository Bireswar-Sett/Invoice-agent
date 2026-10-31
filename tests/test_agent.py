import copy

import pytest

from invoice_agent import create_app, db


TOKEN_A = "test-user-token-A"
TOKEN_B = "test-user-token-B"


def fake_decider(packages, policy_revision):
    output = {}
    for package in packages:
        pid = package["packageId"]
        output[pid] = {
            "packageId": pid,
            "action": "settle_invoice",
            "facts": {
                "vendorName": "Example Vendor",
                "invoiceNumber": f"INV-{pid}",
                "amountMinor": 12500,
                "currency": "INR",
            },
            "evidenceRefs": ["[INV-1]", "[PO-1]", "[PAY-1]"],
            "rationale": (
                "settle_invoice is appropriate because the invoice and purchase order agree, "
                "and the payment record confirms it is unpaid. See [INV-1] and [PO-1] "
                "with [PAY-1] as corroborating evidence."
            ),
        }
    return output


@pytest.fixture()
def client(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite3'}",
        "AI_DECIDER": fake_decider,
        "MAX_PACKAGES_PER_BATCH": 60,
    })
    with app.app_context():
        db.drop_all()
        db.create_all()
    return app.test_client()


def headers(token=TOKEN_A, version="1.0"):
    result = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/a2a+json",
    }
    if version is not None:
        result["A2A-Version"] = version
    return result


def package(pid):
    return {
        "packageId": pid,
        "documents": [
            "[INV-1] Vendor Example Vendor invoice INV-" + pid + " for INR 125.00.",
            "[PO-1] Purchase order confirms the same vendor and INR 125.00.",
            "[PAY-1] Payment ledger shows this invoice has not been paid.",
        ],
    }


def initial_message(packages=None, message_id="msg-1"):
    packages = packages or [package("pkg-1"), package("pkg-2")]
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{
                "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
                "data": {
                    "batchId": "batch-1",
                    "policyRevision": "policy-v1",
                    "packages": packages,
                },
            }],
        },
        "configuration": {
            "returnImmediately": False,
            "historyLength": 20,
            "acceptedOutputModes": [
                "application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json",
            ],
        },
    }


def send_initial(client, token=TOKEN_A, payload=None):
    return client.post(
        "/a2a/message:send",
        json=payload or initial_message(),
        headers=headers(token),
    )


def test_agent_card_is_public_and_has_exact_interface(client):
    response = client.get("/.well-known/agent-card.json")
    assert response.status_code == 200
    card = response.get_json()
    assert card["name"]
    assert card["description"]
    assert card["version"]
    assert isinstance(card["capabilities"], dict)
    assert card["supportedInterfaces"][0]["url"].endswith("/a2a/")
    assert card["supportedInterfaces"][0]["protocolBinding"] == "HTTP+JSON"
    assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
    assert "application/vnd.ga5.invoice-action-proposals+json" in card["defaultOutputModes"]


def test_authentication_and_version(client):
    assert client.get("/a2a/tasks").status_code == 401
    assert client.get("/a2a/tasks", headers=headers(version="2.0")).status_code == 400
    assert client.get("/a2a/tasks", headers=headers(version=None)).status_code == 400


def test_initial_request_returns_input_required_proposals_and_history(client):
    response = send_initial(client)
    assert response.status_code == 200
    task = response.get_json()["task"]
    assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(task["history"]) == 1
    assert len(task["artifacts"]) == 1
    part = task["artifacts"][0]["parts"][0]
    assert part["mediaType"] == "application/vnd.ga5.invoice-action-proposals+json"
    assert len(part["data"]["proposals"]) == 2
    assert all(len(p["evidenceRefs"]) == 3 for p in part["data"]["proposals"])


def test_message_replay_returns_same_task_and_ignores_configuration(client):
    payload = initial_message()
    first = send_initial(client, payload=payload).get_json()["task"]
    changed_config = copy.deepcopy(payload)
    changed_config["configuration"]["returnImmediately"] = True
    second = send_initial(client, payload=changed_config).get_json()["task"]
    assert second == first


def test_same_message_id_with_changed_message_returns_conflict(client):
    payload = initial_message()
    assert send_initial(client, payload=payload).status_code == 200
    changed = copy.deepcopy(payload)
    changed["message"]["parts"][0]["data"]["batchId"] = "different-batch"
    response = send_initial(client, payload=changed)
    assert response.status_code == 409
    assert response.get_json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_results_complete_task_and_only_accepted_results_become_executions(client):
    response = send_initial(client)
    task = response.get_json()["task"]
    proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
    continuation = {
        "message": {
            "messageId": "msg-2",
            "taskId": task["id"],
            "contextId": task["contextId"],
            "role": "ROLE_USER",
            "parts": [{
                "mediaType": "application/vnd.ga5.invoice-action-results+json",
                "data": {
                    "batchId": "batch-1",
                    "results": [
                        {
                            "packageId": proposals[0]["packageId"],
                            "actionId": proposals[0]["actionId"],
                            "action": proposals[0]["action"],
                            "outcome": "ACCEPTED",
                            "receiptNonce": "nonce-accepted-1",
                        },
                        {
                            "packageId": proposals[1]["packageId"],
                            "actionId": proposals[1]["actionId"],
                            "action": proposals[1]["action"],
                            "outcome": "REJECTED",
                            "receiptNonce": "nonce-rejected-2",
                        },
                    ],
                },
            }],
        }
    }
    result = client.post("/a2a/message:send", json=continuation, headers=headers())
    assert result.status_code == 200
    finished = result.get_json()["task"]
    assert finished["status"]["state"] == "TASK_STATE_COMPLETED"
    assert len(finished["history"]) == 2
    assert len(finished["artifacts"]) == 2
    receipt = finished["artifacts"][1]["parts"][0]
    assert receipt["mediaType"] == "application/vnd.ga5.invoice-action-receipts+json"
    executions = receipt["data"]["executions"]
    assert len(executions) == 1
    assert executions[0]["receiptNonce"] == "nonce-accepted-1"

    # Exact continuation replay is safe and returns the complete original Task.
    replay = client.post("/a2a/message:send", json=continuation, headers=headers())
    assert replay.status_code == 200
    assert replay.get_json()["task"] == finished


def test_invalid_action_binding_does_not_complete_task(client):
    task = send_initial(client).get_json()["task"]
    proposals = task["artifacts"][0]["parts"][0]["data"]["proposals"]
    altered = copy.deepcopy(proposals[0])
    altered["action"] = "open_exception"
    continuation = {
        "message": {
            "messageId": "msg-invalid",
            "taskId": task["id"],
            "contextId": task["contextId"],
            "role": "ROLE_USER",
            "parts": [{
                "mediaType": "application/vnd.ga5.invoice-action-results+json",
                "data": {
                    "batchId": "batch-1",
                    "results": [
                        {
                            "packageId": proposals[0]["packageId"],
                            "actionId": proposals[0]["actionId"],
                            "action": "open_exception",
                            "outcome": "ACCEPTED",
                            "receiptNonce": "nonce-1",
                        },
                        {
                            "packageId": proposals[1]["packageId"],
                            "actionId": proposals[1]["actionId"],
                            "action": proposals[1]["action"],
                            "outcome": "REJECTED",
                            "receiptNonce": "nonce-2",
                        },
                    ],
                },
            }],
        }
    }
    response = client.post("/a2a/message:send", json=continuation, headers=headers())
    assert response.status_code == 400
    reread = client.get(f"/a2a/tasks/{task['id']}", headers=headers()).get_json()
    assert reread["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert len(reread["artifacts"]) == 1


def test_user_isolation_and_owner_scoped_listing(client):
    task = send_initial(client, token=TOKEN_A).get_json()["task"]
    outsider_read = client.get(f"/a2a/tasks/{task['id']}", headers=headers(TOKEN_B))
    assert outsider_read.status_code == 404
    outsider_cancel = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=headers(TOKEN_B))
    assert outsider_cancel.status_code == 404
    assert client.get("/a2a/tasks", headers=headers(TOKEN_B)).get_json() == {"tasks": []}
    assert len(client.get("/a2a/tasks", headers=headers(TOKEN_A)).get_json()["tasks"]) == 1


def test_cancel_is_terminal_and_has_no_receipt_artifact(client):
    task = send_initial(client).get_json()["task"]
    response = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=headers())
    assert response.status_code == 200
    canceled = response.get_json()
    assert canceled["status"]["state"] == "TASK_STATE_CANCELED"
    assert len(canceled["artifacts"]) == 1
    again = client.post(f"/a2a/tasks/{task['id']}:cancel", headers=headers())
    assert again.status_code == 409


def test_semantic_cache_ignores_package_id_namespace_and_regenerates_action_ids(client):
    calls = []

    def counting_decider(packages, policy_revision):
        calls.append([p["packageId"] for p in packages])
        return fake_decider(packages, policy_revision)

    client.application.config["AI_DECIDER"] = counting_decider
    original_packages = [package("pkg-1"), package("pkg-2")]
    first = send_initial(
        client,
        payload=initial_message(original_packages, message_id="cache-msg-1"),
    ).get_json()["task"]

    renamed_packages = copy.deepcopy(original_packages)
    renamed_packages[0]["packageId"] = "new-namespace-pkg-1"
    renamed_packages[1]["packageId"] = "new-namespace-pkg-2"
    second_payload = initial_message(renamed_packages, message_id="cache-msg-2")
    second_payload["message"]["parts"][0]["data"]["batchId"] = "batch-2"
    second = send_initial(client, payload=second_payload).get_json()["task"]

    assert len(calls) == 1
    first_proposals = first["artifacts"][0]["parts"][0]["data"]["proposals"]
    second_proposals = second["artifacts"][0]["parts"][0]["data"]["proposals"]
    assert [p["packageId"] for p in second_proposals] == [
        "new-namespace-pkg-1", "new-namespace-pkg-2"
    ]
    assert {p["actionId"] for p in first_proposals}.isdisjoint(
        {p["actionId"] for p in second_proposals}
    )
