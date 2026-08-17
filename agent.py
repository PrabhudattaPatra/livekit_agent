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
            # Speculatively kicks off LLM+TTS as soon as a transcript arrives, instead of
            # waiting out the full min_endpointing_delay (800ms) turn-confirmation pause
            # first, overlapping that wait with generation instead of paying both serially.
            # Kept on after switching the main LLM to Sarvam (see langgraph_agent.py) --
            # still a free latency win even with Sarvam's faster TTFT. Tradeoff: if the
            # user keeps talking past a non-final transcript, the speculative call is
            # wasted (extra Sarvam spend) -- LiveKit's own turn-taking cancels/redoes it
            # automatically, so this is a cost tradeoff, not a correctness risk.
            preemptive_generation=True,
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

        # Voice-pipeline latency SLA logging. AgentSession already computes these numbers
        # internally (per-component metrics from the STT/LLM/TTS plugins, plus VAD-based
        # end-of-utterance timing) -- this just surfaces them per turn so regressions are
        # visible in the logs instead of only being felt as "the agent feels slow". Targets:
        #   STT effective latency (via EOU transcription_delay, see caveat below): 100-150ms
        #   LLM time-to-first-token:                                              150-250ms
        #   TTS time-to-first-audio-byte:                                         75-150ms
        def _on_metrics_collected(ev):
            m = ev.metrics
            if m.type == "eou_metrics":
                # transcription_delay = last_final_transcript_time - last_speaking_time,
                # where last_speaking_time comes from a VAD "stopped speaking" event
                # (livekit-agents' audio_recognition.py). turn_detection="stt" (below) has
                # no separate VAD driving that timestamp, so this is *always* 0 in this
                # config -- not evidence STT is fast, just this metric not applying here.
                # Live-tested and confirmed: two real turns both read exactly 0ms.
                #
                # end_of_utterance_delay isn't a substitute ASR-speed number either: it's
                # time-since-last-speech until the turn is committed, intentionally floored
                # by min_endpointing_delay (0.8s here) -- the deliberate "make sure they're
                # done talking" wait, not STT latency. Scoring it against the 100-150ms
                # target would read "SLOW" on every single turn by construction, which is
                # the same false signal in the other direction. Log it unscored, for
                # context only.
                eou_ms = m.end_of_utterance_delay * 1000
                if m.transcription_delay > 0:
                    ms = m.transcription_delay * 1000
                    status = "OK" if ms <= 150 else "SLOW"
                    logger.info(
                        "[SLA] STT transcription_delay=%.0fms (target 100-150ms) [%s]",
                        ms, status,
                    )
                else:
                    logger.info(
                        "[SLA] STT transcription_delay=n/a (turn_detection='stt' has no "
                        "separate VAD timing) | end_of_utterance_delay=%.0fms "
                        "(includes the configured min_endpointing_delay wait, not scored)",
                        eou_ms,
                    )
            elif m.type == "llm_metrics":
                ms = m.ttft * 1000
                status = "OK" if ms <= 250 else "SLOW"
                logger.info("[SLA] LLM ttft=%.0fms (target 150-250ms) [%s]", ms, status)
            elif m.type == "tts_metrics":
                ms = m.ttfb * 1000
                status = "OK" if ms <= 150 else "SLOW"
                logger.info("[SLA] TTS ttfb=%.0fms (target 75-150ms) [%s]", ms, status)
            elif m.type == "stt_metrics" and not m.streamed:
                # Only meaningful for non-streaming STT -- duration is 0.0 while streaming
                # (which is what this agent uses), so skip logging a always-zero number.
                logger.info("[SLA] STT duration=%.0fms (non-streaming)", m.duration * 1000)

        session.on("metrics_collected", _on_metrics_collected)

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