import { NextResponse } from 'next/server';

// Lightweight liveness check -- deliberately does NOT touch LiveKit or any other
// external dependency (unlike /api/token). This just confirms the Next.js server
// process itself is up and serving requests, for Render's health check / uptime pings.
export async function GET() {
  return NextResponse.json({ status: 'ok' });
}
