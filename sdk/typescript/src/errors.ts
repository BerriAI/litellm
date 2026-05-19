/** Typed errors raised by XctClient. */

export class XctError extends Error {
  status?: number;
  body?: unknown;

  constructor(message: string, status?: number, body?: unknown) {
    super(message);
    this.name = "XctError";
    this.status = status;
    this.body = body;
  }
}

export class AuthError extends XctError {
  constructor(message: string, status?: number, body?: unknown) {
    super(message, status, body);
    this.name = "AuthError";
  }
}

export class RateLimitError extends XctError {
  constructor(message: string, status?: number, body?: unknown) {
    super(message, status, body);
    this.name = "RateLimitError";
  }
}

export class CapabilityNotFoundError extends XctError {
  constructor(message: string, status?: number, body?: unknown) {
    super(message, status, body);
    this.name = "CapabilityNotFoundError";
  }
}

export function fromResponse(status: number, body: unknown): XctError {
  const msg = extractMessage(body) || `HTTP ${status}`;
  if (status === 401 || status === 403) return new AuthError(msg, status, body);
  if (status === 404) return new CapabilityNotFoundError(msg, status, body);
  if (status === 429) return new RateLimitError(msg, status, body);
  return new XctError(msg, status, body);
}

function extractMessage(body: unknown): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as Record<string, unknown>).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0];
      if (first && typeof first === "object" && "msg" in first) {
        return String((first as Record<string, unknown>).msg);
      }
    }
  }
  if (body && typeof body === "object" && "error" in body) {
    const err = (body as Record<string, unknown>).error;
    return typeof err === "string" ? err : String(err);
  }
  return "";
}
