# A2A Invoice Action Agent

A Flask implementation of the assignment's A2A 1.0 HTTP+JSON surface. It is designed for Render and uses PostgreSQL for durable task, idempotency, decision-cache, and execution state.

## What is implemented

- Public `GET /.well-known/agent-card.json` discovery card.
- Protected A2A routes under `/a2a/`:
  - `POST /message:send`
  - `GET /tasks/{id}`
  - `GET /tasks`
  - `POST /tasks/{id}:cancel`
- Required `Authorization: Bearer ...`, `A2A-Version: 1.0`, and `application/a2a+json` request media type for POST endpoints.
- Per-token principal isolation. The raw Bearer token is never stored; a SHA-256 digest is used as the principal key.
- Initial invoice batches produce one typed proposal per package and a `TASK_STATE_INPUT_REQUIRED` Task. The initial history message is retained.
- Configurable OpenAI-compatible model call. Uncached packages are sent together in one batch request.
- Server-side checks for proposal package IDs, action enum, invoice facts, three exact bracketed evidence references, rationale length, and rationale citations.
- Canonical package-content decision cache (including policy revision, excluding the top-level package ID namespace). The action ID is regenerated for each task/batch namespace.
- Continuation validation against the stored Task, context, batch, package, action ID, and action.
- Only `ACCEPTED` results become executions. The receipt nonce is copied exactly from the grader; the model never generates it.
- Proposal artifact retained, plus one receipt artifact on a valid continuation (including an empty executions array if every proposal is rejected).
- Per-principal message idempotency with recursively key-sorted compact JSON hashing of the `message` only. `configuration` is ignored.
- Unique DB constraints for idempotency keys and action IDs.
- Atomic task state transitions so cancellation and result completion cannot both win. Terminal Tasks are not modified afterward.
- Owner-scoped task reads, task lists, continuations, and cancellation.
- Response JSON is compacted and checked against the 512 KiB limit.
- Health endpoint at `/healthz`.

## Important integration note

The assignment description does not include the actual grader package JSON schema or the hidden invoice documents. This project therefore treats each package as an opaque JSON object, identifies its package ID from `packageId` (also accepts `package_id` or `id`), and sends the original package content to the model. The built-in prompt explicitly handles archive examples, negation, duplicate-invoice distinction, approval authority, and evidence selection.

The LLM's output is validated before it is accepted. This cannot guarantee the model will reason correctly on unseen documents; model choice, prompt tuning, and evidence quality remain important. Do not use the demo fake decider in production.

## 1. Deploy on Render with a Blueprint

1. Extract this ZIP.
2. Push the extracted folder to a new GitHub repository.
3. In Render, choose **New → Blueprint** and connect that repository.
4. Render reads `render.yaml` and creates the web service plus a Free Postgres database.
5. During Blueprint creation, provide `AI_API_KEY` for an OpenAI-compatible model provider.
6. After deployment, open the web service's `onrender.com` URL and check `/healthz`.
7. Your submitted A2A base URL is:

   `https://YOUR-SERVICE.onrender.com/a2a/`

The Agent Card is at the origin-level path:

`https://YOUR-SERVICE.onrender.com/.well-known/agent-card.json`

Render's free web service can spin down when idle, and its filesystem is ephemeral. The Blueprint uses Postgres rather than SQLite for task persistence. Render's Free Postgres currently expires 30 days after creation, has a 1 GB limit, and has no backups; upgrade the database before that deadline if the task state must survive longer. See Render's free-tier documentation: https://render.com/docs/free.

For a longer-lived deployment, change the database `plan` in `render.yaml` to a paid Postgres plan and choose a paid web plan if you need to avoid free-service spin-down.

## 2. Required environment variables

| Variable | Default | Meaning |
|---|---|---|
| `A2A_BASE_PATH` | `/a2a` | URL path prefix for the four A2A service routes |
| `PUBLIC_BASE_URL` | Derived from Render | Exact submitted base URL, including `/a2a/`; set explicitly if using a custom domain or different path |
| `DATABASE_URL` | Local SQLite | SQLAlchemy connection string; use Render Postgres in production |
| `AI_BASE_URL` | `https://aipipe.org/openai/v1` | AI Pipe OpenAI-compatible API base URL |
| `AI_API_KEY` | empty | AI Pipe token; required before processing invoice batches |
| `AI_MODEL` | `gpt-5-nano` | Model name supported by AI Pipe OpenAI route |

For AI Pipe, use its OpenAI-compatible Chat Completions route: `https://aipipe.org/openai/v1/chat/completions`. The application appends `/chat/completions` to `AI_BASE_URL` automatically. The `gpt-5-nano` model was verified with this route. The request uses `max_completion_tokens` and deliberately omits `temperature`: GPT-5 reasoning models reject custom temperature values.
| `AI_TIMEOUT_SECONDS` | `35` | LLM request timeout; keep within the grader's request budget |
| `AI_MAX_OUTPUT_TOKENS` | `12000` | Maximum generated proposal output |
| `MAX_PACKAGES_PER_BATCH` | `60` | Safety limit on packages in a single batch |
| `MAX_REQUEST_BYTES` | `2097152` | Maximum HTTP request body size |
| `MAX_RESPONSE_BYTES` | `524288` | Maximum successful response size (512 KiB) |

`AI_BASE_URL` must expose the OpenAI-compatible `/chat/completions` endpoint and support JSON-object response mode. Examples include OpenAI and compatible hosted providers. For a different provider protocol, adapt `invoice_agent/ai.py`.

## 3. Local development

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Set at least `AI_API_KEY` in `.env`, then run:

```bash
python run.py
```

The local service is available at `http://127.0.0.1:8000`. To exercise the protected routes locally, send a Bearer token of your choice. The token string is treated as the principal identity; use the same exact token for all requests belonging to one user and different tokens for different users.

The local SQLite database is stored under `.data/`. SQLite is appropriate for local tests, not production concurrency. Render is configured to use Postgres.

## 4. Example requests

Agent Card (public):

```bash
curl https://YOUR-SERVICE.onrender.com/.well-known/agent-card.json
```

List tasks (protected):

```bash
curl -H 'Authorization: Bearer YOUR_USER_TOKEN' \\
     -H 'A2A-Version: 1.0' \\
     https://YOUR-SERVICE.onrender.com/a2a/tasks
```

Initial message envelope (the package data is abbreviated):

```bash
curl -X POST https://YOUR-SERVICE.onrender.com/a2a/message:send \\
  -H 'Authorization: Bearer YOUR_USER_TOKEN' \\
  -H 'A2A-Version: 1.0' \\
  -H 'Content-Type: application/a2a+json' \\
  -d '{
    "message": {
      "messageId": "msg-unique-001",
      "role": "ROLE_USER",
      "parts": [{
        "mediaType": "application/vnd.ga5.invoice-claim-batch+json",
        "data": {
          "batchId": "batch-001",
          "policyRevision": "policy-v1",
          "packages": []
        }
      }]
    },
    "configuration": {
      "returnImmediately": false,
      "historyLength": 20,
      "acceptedOutputModes": [
        "application/vnd.ga5.invoice-action-proposals+json",
        "application/vnd.ga5.invoice-action-receipts+json"
      ]
    }
  }'
```

The real `packages` array must be non-empty. Save the returned `task.id`, `task.contextId`, and proposal artifact. The grader will later send the result continuation to the same `/message:send` route with the same Task and Context IDs.

## 5. Tests

Run the included automated tests:

```bash
pytest -q
```

The test suite uses an injected fake decision function and does not spend model tokens. It checks the public card, authentication/version handling, proposal lifecycle, message replay/conflict, result binding, accepted-only receipts, user isolation, and cancellation.

Before using the grader, add integration tests against the actual case package format once you have inspected a sample request. The current test fixtures are synthetic examples only.

## 6. Security and operational notes

- The agent accepts any syntactically valid Bearer token as a distinct principal because the assignment defines identity by the exact token presented. In this challenge setting, tokens are opaque user identifiers. If you need a production identity provider, replace `principal_from_bearer` with verified token introspection/JWT validation while preserving one-principal-per-token semantics.
- Never log or persist raw Bearer tokens, provider API keys, invoice files, or receipt nonces unnecessarily.
- All invoice documents are passed as untrusted input to the model. The prompt tells the model to ignore embedded instructions. This is defense-in-depth, not a substitute for deterministic validation.
- A proposal is not an execution authorization. Only a matched `ACCEPTED` continuation creates an `ExecutionRecord` and receipt artifact.
- This challenge has no external payment connector. `ExecutionRecord` is the durable record of the approved business action; it deliberately does not initiate a real bank transfer.
- Use Postgres in Render. Free web-service local files are ephemeral, and free web services cannot use persistent disks.


## 7. Diagnostics without using the grader

The `scripts/diagnose.py` tool checks the live/local HTTP surface using only harmless probes. It does **not** submit a valid invoice batch, create a Task, execute a business action, or consume a grader Check/Save.

Run the public Render checks:

```bash
python scripts/diagnose.py --base-url https://YOUR-SERVICE.onrender.com
```

Run local application checks after starting `python run.py` in another terminal:

```bash
python scripts/diagnose.py --base-url http://127.0.0.1:8000
```

Run a separate tiny AI Pipe JSON-mode smoke test (this uses a small model request and may consume a small amount of AI Pipe quota):

```bash
export AI_API_KEY='your-aipipe-token'
export AI_BASE_URL='https://aipipe.org/openai/v1'
export AI_MODEL='gpt-5-nano'
python scripts/diagnose.py --base-url https://YOUR-SERVICE.onrender.com --test-ai
```

Do not paste the token into chat or commit it to Git. The script never prints the token.

### Collect Render logs

Install the Render CLI from https://render.com/docs/cli, then authenticate once:

```bash
render login
render services
```

Find the service ID for `a2a-invoice-agent` (it starts with `srv-`) and collect recent runtime, error, handler, and available HTTP request logs:

```bash
RENDER_SERVICE_ID=srv-xxxxxxxx ./scripts/collect_render_logs.sh
```

The logs are written under `./diagnostics/`. Inspect them for credentials or private invoice contents before sharing. Render CLI supports resource, time, level, method, path and status-code filters; request logs may depend on workspace plan. See https://render.com/docs/cli-reference and https://render.com/docs/logging.

The provider integration now logs the provider host, model, package count, request size, latency, HTTP status, and a short redacted provider error body. It never logs the API key or invoice prompt. Redeploy this code revision before expecting the improved provider diagnostics in Render logs.
