import { AccessToken, AgentDispatchClient } from 'livekit-server-sdk';
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    // You can pass an optional name, or we generate a random one
    const participantName = searchParams.get('name') || `User_${Math.floor(Math.random() * 1000)}`;

    // A unique room per caller -- concurrent visitors must not share a room (they'd hear
    // each other's audio and each other's agent session, and each /api/token call would
    // dispatch another agent into the same room on top of it).
    const roomName = `InteriorDesignAgentRoom-${crypto.randomUUID()}`;

    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const wsUrl = process.env.LIVEKIT_URL;

    if (!apiKey || !apiSecret || !wsUrl) {
      return NextResponse.json(
        { error: 'Server misconfigured. LiveKit credentials missing.' },
        { status: 500 }
      );
    }

    const at = new AccessToken(apiKey, apiSecret, {
      identity: participantName,
      // Token expires in 10 minutes
      ttl: '10m',
    });

    at.addGrant({ roomJoin: true, room: roomName });

    // Explicitly dispatch the agent to the room so it joins when the user connects
    const dispatchClient = new AgentDispatchClient(wsUrl, apiKey, apiSecret);
    try {
      await dispatchClient.createDispatch(roomName, 'InteriorDesignAgent');
      console.log('Agent dispatched successfully');
    } catch (dispatchErr) {
      console.log('Agent dispatch ignored or failed (might already be dispatched):', dispatchErr);
    }

    return NextResponse.json({ token: await at.toJwt() });
  } catch (error) {
    console.error('Error generating token:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
