import { describe, expect, it } from "vitest";

import {
  chatGPTStartRequest,
  initialSignInFlowState,
  signInFailureMessage,
  signInFlowReducer,
  type SignInFlowState,
} from "./chatgpt_sign_in_flow";

const started = (verificationUri = "https://auth.openai.com/codex/device"): SignInFlowState =>
  signInFlowReducer(initialSignInFlowState, {
    type: "started",
    response: {
      user_code: "WXYZ-9876",
      verification_uri: verificationUri,
      expires_in: 900,
      interval: 5,
      flow_handle: "fh-1",
    },
    now: 1_000,
  });

const connected = (): SignInFlowState =>
  signInFlowReducer(started(), { type: "polled", response: { status: "connected", account_id: "acct-1" } });

describe("signInFlowReducer", () => {
  it("moves from starting to awaiting with the code, expiry, interval and handle from the start response", () => {
    const expected: SignInFlowState = {
      kind: "awaiting",
      attempt: 0,
      pollCount: 0,
      userCode: "WXYZ-9876",
      verificationUri: "https://auth.openai.com/codex/device",
      expiresAt: 901_000,
      intervalSeconds: 5,
      flowHandle: "fh-1",
    };
    expect(started()).toEqual(expected);
  });

  it.each([
    "https://evil.example/codex/device",
    "http://auth.openai.com/codex/device",
    "https://auth.openai.com.evil.example/codex/device",
    "https://user@auth.openai.com/codex/device",
    "https://auth.openai.com:8443/codex/device",
    "not a url",
  ])("falls back to the ChatGPT device page for unsafe verification URL %s", (uri) => {
    const state = started(uri);
    expect(state.kind === "awaiting" && state.verificationUri).toBe("https://auth.openai.com/codex/device");
  });

  it("keeps waiting on pending as a new state so the next poll is scheduled", () => {
    const awaiting = started();
    const pending = signInFlowReducer(awaiting, { type: "polled", response: { status: "pending", account_id: null } });
    expect(pending).not.toBe(awaiting);
    expect(pending).toEqual({ ...awaiting, pollCount: 1 });
  });

  it("connects with the account id from the poll", () => {
    expect(connected()).toEqual({ kind: "connected", attempt: 0, accountId: "acct-1" });
  });

  it("expires locally once the code lifetime has passed", () => {
    const expired = signInFlowReducer(started(), { type: "timed_out" });
    expect(expired).toEqual({ kind: "failed", attempt: 0, reason: "expired" });
    expect(expired.kind === "failed" && signInFailureMessage(expired)).toContain("expired");
  });

  it("fails with the backend's message when a poll is rejected, then restarts with a new attempt", () => {
    const failed = signInFlowReducer(started(), { type: "errored", message: "Invalid or expired flow handle" });
    const expected: SignInFlowState = {
      kind: "failed",
      attempt: 0,
      reason: "error",
      message: "Invalid or expired flow handle",
    };
    expect(failed).toEqual(expected);
    expect(failed.kind === "failed" && signInFailureMessage(failed)).toBe("Invalid or expired flow handle");
    expect(signInFlowReducer(failed, { type: "restarted" })).toEqual({ kind: "starting", attempt: 1 });
  });

  it("ignores a late poll result or error after the flow connected", () => {
    const done = connected();
    expect(signInFlowReducer(done, { type: "polled", response: { status: "pending", account_id: null } })).toBe(done);
    expect(signInFlowReducer(done, { type: "errored", message: "late" })).toBe(done);
    expect(signInFlowReducer(done, { type: "timed_out" })).toBe(done);
  });

  it("ignores a second start response once the flow is already awaiting", () => {
    const awaiting = started();
    const again = signInFlowReducer(awaiting, {
      type: "started",
      response: { user_code: "OTHER", verification_uri: "", expires_in: 1, interval: 1, flow_handle: "fh-2" },
      now: 0,
    });
    expect(again).toBe(awaiting);
  });
});

describe("chatGPTStartRequest", () => {
  it("sets only the team id for a team owner", () => {
    expect(chatGPTStartRequest("team", "team-1", "")).toEqual({
      ownerTeamId: "team-1",
      ownerKeyHash: null,
      displayName: null,
    });
  });

  it("sets only the key hash for a virtual key owner and trims the display name", () => {
    expect(chatGPTStartRequest("key", "hash-1", "  Shared ChatGPT  ")).toEqual({
      ownerTeamId: null,
      ownerKeyHash: "hash-1",
      displayName: "Shared ChatGPT",
    });
  });
});
