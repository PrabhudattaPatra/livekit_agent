"use client";

import { useState, useCallback } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  BarVisualizer,
  VoiceAssistantControlBar,
  useVoiceAssistant,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { Mic } from "lucide-react";

export default function Home() {
  const [token, setToken] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startConversation = async () => {
    try {
      setConnecting(true);
      setError(null);
      // Fetch a new token from our backend API
      const res = await fetch("/api/token");
      if (!res.ok) {
        throw new Error("Failed to fetch token");
      }
      const data = await res.json();
      setToken(data.token);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred");
    } finally {
      setConnecting(false);
    }
  };

  const handleDisconnect = useCallback(() => {
    setToken(null);
  }, []);

  return (
    <main className="app-container">
      {!token ? (
        <div className="landing-view">
          <div className="hero-section">
            <h1 className="title">AI Interior Design Assistant</h1>
            <p className="subtitle">
              Talk through your space, get design ideas, and book a consultation with a professional — all by voice.
            </p>
            
            <button
              onClick={startConversation}
              disabled={connecting}
              className={`start-btn ${connecting ? "connecting" : ""}`}
            >
              {connecting ? (
                <div className="loader"></div>
              ) : (
                <>
                  <Mic size={24} />
                  <span>Start Conversation</span>
                </>
              )}
            </button>
            {error && <p className="error-text">{error}</p>}
          </div>
        </div>
      ) : (
        <LiveKitRoom
          serverUrl={process.env.NEXT_PUBLIC_LIVEKIT_URL}
          token={token}
          connect={true}
          audio={true}
          video={false}
          onDisconnected={handleDisconnect}
          className="livekit-room-container"
        >
          <AgentUI />
          <RoomAudioRenderer />
        </LiveKitRoom>
      )}
    </main>
  );
}

// A sub-component to render the active agent UI
function AgentUI() {
  const { state, audioTrack } = useVoiceAssistant();

  return (
    <div className="active-agent-view">
      <div className="status-indicator">
        <div className={`pulse-ring ${state}`}></div>
        <p className="status-text">
          {state === "connecting" && "Connecting..."}
          {state === "initializing" && "Waking up..."}
          {state === "listening" && "Listening..."}
          {state === "speaking" && "Agent is speaking"}
          {state === "thinking" && "Thinking..."}
          {state === "disconnected" && "Disconnected"}
        </p>
      </div>

      <div className="visualizer-container">
        {audioTrack && (
          <BarVisualizer
            state={state}
            barCount={5}
            trackRef={audioTrack}
            className="agent-visualizer"
          />
        )}
      </div>

      <div className="control-bar-container">
         <VoiceAssistantControlBar />
      </div>
    </div>
  );
}
