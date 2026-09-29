// ---------------------------------------------------------------------------
// SessionStore — per-session mutable state keyed by MCP session id.
//
// On streamable HTTP transport, each MCP session carries an `Mcp-Session-Id`
// header; we use that as the key. On stdio transport, one process serves one
// client, so we use the fixed sentinel `"stdio"`.
//
// Phase 1: only `defaultAccountAlias` is tracked. Future phases may add
// ephemeral per-session preferences (read fanout opt-out, etc.).
// ---------------------------------------------------------------------------

import { SessionState } from './types.js';

export const STDIO_SESSION_ID = 'stdio';

export class SessionStore {
  private sessions = new Map<string, SessionState>();

  getOrCreate(sessionId: string): SessionState {
    let s = this.sessions.get(sessionId);
    if (!s) {
      s = { sessionId, createdAt: Date.now() };
      this.sessions.set(sessionId, s);
    }
    return s;
  }

  get(sessionId: string): SessionState | undefined {
    return this.sessions.get(sessionId);
  }

  delete(sessionId: string): void {
    this.sessions.delete(sessionId);
  }

  size(): number {
    return this.sessions.size;
  }
}
