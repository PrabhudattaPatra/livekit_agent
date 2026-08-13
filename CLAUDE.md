# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AI Interior Design voice assistant, delivered over a LiveKit voice session. It holds a natural
discovery conversation about the user's home (room type/dimensions, style, budget, furniture,
colors, lifestyle, renovation goals), offers general design guidance along the way, and — when it
detects the user needs personalized/professional expertise — offers to book a consultation with a
professional interior designer, scheduled on Google Calendar. Speech goes in via Sarvam STT, is
routed through a LangGraph agent (tool-calling LLM + Google Calendar tools used for booking
consultations), and comes back out via Sarvam TTS. A Next.js frontend in `frontend/` is the browser
client that joins the LiveKit room; it has its own `CLAUDE.md`/`AGENTS.md` (Next.js-specific rules)
— read that when working inside `frontend/`.

## Commands

Backend uses `uv` (there's a `uv.lock`); `requirements.txt`/`pip` also works and is what the Docker
image uses.

```bash
# install deps
uv sync                      # or: pip install -r requirements.txt

# run the agent (LiveKit CLI worker modes)
python agent.py dev          # local dev, connects to LiveKit and auto-reloads
python agent.py start        # production mode (what the Dockerfile CMD runs)

# regenerate Google OAuth token.json after it expires
python reauth_google.py
```

Frontend (`frontend/`), Node 20+, standard Next.js scripts:

```bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
```

There is no test suite in this repo currently.

## Environment / credentials

Configured via `.env` (loaded with `python-dotenv`). Required for the backend to run:
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, Sarvam key, Groq key (used only by the
small `ack_llm` model, see below), a Portkey API key (`PORTKEY_API_KEY`, used to route the main
LLM), Google Cloud credentials, and optionally `OTEL_EXPORTER_OTLP_ENDPOINT`/
`OTEL_EXPORTER_OTLP_HEADERS` for LangSmith tracing.

Google Calendar auth is file-based (`credentials.json` + `token.json`, both gitignored). Locally,
run `reauth_google.py` once to produce `token.json`. In deployment (Railway), these files don't
exist in the image, so `agent.py`'s `setup_google_credentials()` reconstructs them at startup from
the `GOOGLE_CREDENTIALS_JSON` / `GOOGLE_TOKEN_JSON` env vars — keep that in mind if you change how
credentials are loaded, both code paths need to keep working.

`DATABASE_URL`, if set, enables a Postgres-backed LangGraph checkpointer (conversation memory across
turns/reconnects via `thread_id` = session UUID). Without it, the graph runs with no persistence.

## Architecture

**`agent.py`** — LiveKit entry point. On import it wires up OpenTelemetry → LangSmith tracing
(`langsmith_processor.py`) and reconstructs Google credential files from env vars if present. It
defines `InteriorDesignAgent` (a thin `livekit.agents.Agent` wrapper) and an `entrypoint` function
registered via `cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="InteriorDesignAgent"))`
— that `agent_name` string must exactly match the one `frontend/src/app/api/token/route.ts` passes
to `dispatchClient.createDispatch(...)`, or the frontend will connect to a room the worker never
joins. The entrypoint builds an `AgentSession` with Sarvam STT/TTS and a `langchain.LLMAdapter`
whose `graph` is the compiled LangGraph agent from `langgraph_agent.py`. Session lifetime is tied to
the room: it opens a Postgres checkpointer connection (if `DATABASE_URL` is set) for the duration of
the session and blocks on the room's `disconnected` event before returning.

**`langgraph_agent.py`** — the actual agent logic, structured in numbered parts:
1. `_sanitize_events` — normalizes raw Google Calendar API responses into compact JSON before
   they're handed back to the LLM (keeps token usage down, avoids leaking raw API shape).
2. Custom `BaseTool` subclasses wrapping `langchain-google-community` calendar operations
   (`search_appointments`, `create_appointment`, `search_appointments_by_email`,
   `update_appointment`, `cancel_appointment`). Each pins fixed params (calendar id, timezone
   `Asia/Calcutta`, etc.) and enforces a 15s timeout via `asyncio.wait_for`, returning a
   `"TOOL_ERROR: ..."` string on failure/timeout rather than raising — errors are meant to be
   spoken back to the user, not crash the graph.
3. The main LLM (`response_model`) is routed entirely through **Portkey**
   (`PORTKEY_GATEWAY_URL` + `pc-liveki-1a1e70` config, referenced by ID via `createHeaders`) — it's
   instantiated as `ChatOpenAI` with a dummy API key purely because Portkey's gateway speaks the
   OpenAI wire format; no provider/model is chosen in this file. The actual routing lives in the
   Portkey config itself: `cache.mode: simple`, then a top-level `fallback` strategy with two
   targets — first a `loadbalance` pair (50/50) of Groq-hosted `qwen/qwen3.6-27b` (reasoning
   disabled), then a fallback to Fireworks' `deepseek-v4-flash`. Changing models/providers/weights
   means editing the Portkey config (`pc-liveki-1a1e70`), not this code.
4. `system_prompt_template` — a large voice-specific system prompt: TTS-friendly output rules
   (no markdown/lists/emojis, spell out numbers, Hindi/English code-mixing rules), an open-ended
   design-discovery script (what to ask about, how to keep it conversational rather than an
   interrogation), explicit criteria for when to offer a professional consultation, and
   step-by-step scripts for booking/updating/cancelling that consultation. When changing agent
   behavior — discovery flow, when it offers a consult, or the scheduling scripts — this prompt is
   almost always the place to edit, not the graph wiring.
5. `chat_node` — the LLM node. Deduplicates consecutive identical `AIMessage`s, truncates history
   to the last 10 messages, injects a fresh `current_datetime` into the system prompt every call
   (so the agent's notion of "now" never goes stale), and streams tokens via
   `get_stream_writer()`. It has a heuristic to detect a model emitting raw
   `<function.../<tool...>` tool-call syntax as text (seen with some Groq/Llama models) and stops
   streaming that as spoken content.
6. `tool_node_with_ack` — wraps `ToolNode`. If a tool call is still running after 0.5s, it fires a
   short LLM-generated (`ack_llm`, Groq `llama-3.1-8b-instant`) verbal acknowledgment
   ("One moment, let me check that...") through the stream so the user isn't left in silence
   during slower Calendar API calls; cancelled once the tool resolves.
7. `create_interior_design_graph(checkpointer)` — builds the two-node graph (`chat_node` ⇄
   `tools`, `tools_condition` for routing) and compiles it, optionally with a checkpointer for
   persistence. This is the function `agent.py` calls to get the graph handed to LiveKit's
   `LLMAdapter`.

**`langsmith_processor.py`** — a custom OpenTelemetry `SpanProcessor` that rewrites LiveKit Agents'
spans (STT/LLM/TTS/tool spans) into LangSmith's expected attribute shape so a voice session shows up
as a coherent conversation thread in LangSmith, rather than as disconnected OTel spans.

**`agent.ipynb`** — notebook mirror of the same agent logic, used for interactive iteration outside
the LiveKit runtime; not part of the deployed path.

**`frontend/`** — Next.js app with a single API route (`src/app/api/token/route.ts`) that mints a
LiveKit access token and explicitly dispatches the `InteriorDesignAgent` into a static room
(`InteriorDesignAgentRoom`) before the browser client connects.

## Deployment

Railway, building `Dockerfile` directly (`railway.json` sets `builder: DOCKERFILE`). The Dockerfile
only packages the Python backend (`frontend/` is excluded via `.dockerignore`) and runs
`python agent.py start`. The frontend deploys separately (its own `railway.json` under
`frontend/`, forced to Node 20 / Nixpacks).
