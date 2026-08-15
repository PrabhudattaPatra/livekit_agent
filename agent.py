import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from livekit.agents.beta.tools import EndCallTool
from opentelemetry.sdk.trace import TracerProvider
from livekit.agents.telemetry import set_tracer_provider
from langsmith_processor import LangSmithSpanProcessor

logging.basicConfig(level=logging.INFO)
logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").setLevel(logging.DEBUG)


def setup_langsmith():
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    if not endpoint or not headers:
        print("⚠️ Warning: OTEL environment variables not set. Tracing disabled.")
        return
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(LangSmithSpanProcessor())
    set_tracer_provider(trace_provider)
    print("✅ LangSmith tracing enabled")

setup_langsmith()


def setup_google_credentials():
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    
    if creds_json:
        with open("credentials.json", "w") as f:
            f.write(creds_json)
        print("✅ Reconstructed credentials.json")
        
    if token_json:
        with open("token.json", "w") as f:
            f.write(token_json)
        print("✅ Reconstructed token.json")

setup_google_credentials()

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import sarvam, langchain
from langgraph_agent import create_interior_design_graph

logger = logging.getLogger("interior-design-agent")

class InteriorDesignAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a warm, knowledgeable voice assistant for Aethel Studio, an interior design studio. You help callers think through ideas for their space and, when it's a good fit, help them book a consultation with one of our professional interior designers.""",
            tools=[EndCallTool(
                extra_description="",
                end_instructions="Thank the user for their time and say goodbye.",
                delete_room=True,
            )],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Greet the caller as Aethel Studio's design assistant. Open with something "
                "like \"Welcome to Aethel Studio, how can I help you today?\" -- keep it warm "
                "and brief, then let them respond."
            ),
            allow_interruptions=True,
        )

async def entrypoint(ctx: JobContext):
    # Connect worker process to room explicitly
    await ctx.connect()

    session_id = str(uuid.uuid4())
    DB_URI = os.getenv("DATABASE_URL")
    
    checkpointer = None
    if DB_URI:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(DB_URI)
        checkpointer = await checkpointer_cm.__aenter__()
        await checkpointer.setup()

    try:
        session = AgentSession(
            stt=sarvam.STT(
                language="en-IN",
                model="saaras:v3",
                mode="codemix",
                flush_signal=True,       
            ),
            llm=langchain.LLMAdapter(
                graph=create_interior_design_graph(checkpointer),
                config={"configurable": {"thread_id": session_id}},
                stream_mode="custom",
            ),
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="shubh",
                speech_sample_rate=16000,
                pace=1.0,
            ),
            turn_detection="stt",         
            min_endpointing_delay=0.8,    
            preemptive_generation=False, 
        )

        await session.start(
            agent=InteriorDesignAgent(),
            room=ctx.room,
        )

        # Handle prolonged user silence (right after the greeting or mid-conversation).
        # user_away_timeout (default 15s) fires "user_state_changed" -> "away" once both the
        # user and agent have been idle that long -- but it fires ONLY ONCE per silent
        # stretch: user_state stays "away" and won't re-arm the framework's own timer again
        # until real speech resets it. So the nudge -> (wait) -> goodbye escalation is driven
        # by our own timer here, not by waiting for a second "away" event that will never come
        # during continuous silence. If the user starts speaking at any point, we bail out.
        away_in_progress = False
        user_spoke = asyncio.Event()

        async def _handle_user_away():
            nonlocal away_in_progress
            if away_in_progress:
                return
            away_in_progress = True
            user_spoke.clear()
            try:
                await session.generate_reply(
                    instructions=(
                        "The user has gone quiet for a moment. Check in warmly and briefly — "
                        "for example, ask if they're still there or need a moment to think. Keep it short."
                    )
                )
                try:
                    await asyncio.wait_for(user_spoke.wait(), timeout=15.0)
                    return  # user responded, nothing more to do
                except asyncio.TimeoutError:
                    pass

                await session.generate_reply(
                    instructions=(
                        "The user has stayed silent even after you checked in. Say a brief, warm "
                        "goodbye and mention they're welcome to reach out again anytime."
                    )
                )
                # Don't rely on the LLM to voluntarily call end_call here -- it's not a normal
                # conversational turn, and testing showed it reliably speaks the goodbye but
                # doesn't reliably invoke the tool on its own. Hang up deterministically instead.
                logger.info("ending call: user stayed silent through the away check-in")
                await ctx.delete_room()
            except Exception:
                logger.exception("Failed during away-state handling")
            finally:
                away_in_progress = False

        def _on_user_state_changed(ev):
            if ev.new_state == "speaking":
                user_spoke.set()
            elif ev.new_state == "away":
                asyncio.create_task(_handle_user_away())

        session.on("user_state_changed", _on_user_state_changed)

        background_audio = BackgroundAudioPlayer(
            ambient_sound=AudioConfig(BuiltinAudioClip.OFFICE_AMBIENCE, volume=1.0),
        )
        await background_audio.start(room=ctx.room, agent_session=session)

        # Keep session open until room disconnects
        shutdown_event = asyncio.Event()
        ctx.room.on("disconnected", lambda *args: shutdown_event.set())
        await shutdown_event.wait()

    finally:
        if DB_URI and checkpointer:
            await checkpointer_cm.__aexit__(None, None, None)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="InteriorDesignAgent",
        )
    )