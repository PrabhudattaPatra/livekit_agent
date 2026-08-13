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
            instructions="""You are a warm, knowledgeable voice assistant for an interior design studio. You help callers think through ideas for their space and, when it's a good fit, help them book a consultation with one of our professional interior designers.""",
            tools=[EndCallTool(
                extra_description="",
                end_instructions="Thank the user for their time and say goodbye.",
                delete_room=True,
            )],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user warmly as their interior design assistant, and invite them to tell you about the space they're looking to work on.",
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
                speaker="shubh"
            ),
            turn_detection="stt",         
            min_endpointing_delay=0.8,    
            preemptive_generation=False, 
        )

        await session.start(
            agent=InteriorDesignAgent(),
            room=ctx.room,
        )
        
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