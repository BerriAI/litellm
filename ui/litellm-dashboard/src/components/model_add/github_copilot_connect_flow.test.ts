import { describe, expect, it } from "vitest";

import { connectFlowReducer, type ConnectFlowState, initialConnectFlowState } from "./github_copilot_connect_flow";

const started = (verificationUri = "https://github.com/login/device"): ConnectFlowState =>
  connectFlowReducer(initialConnectFlowState, {
    type: "started",
    response: {
      user_code: "ABCD-1234",
      verification_uri: verificationUri,
      expires_in: 900,
      interval: 5,
      flow_handle: "fh-1",
    },
    now: 1_000,
  });

describe("connectFlowReducer", () => {
  it("moves from starting to awaiting with the code, expiry and interval from the start response", () => {
    const expected: ConnectFlowState = {
      kind: "awaiting",
      attempt: 0,
      pollCount: 0,
      userCode: "ABCD-1234",
      verificationUri: "https://github.com/login/device",
      expiresAt: 901_000,
      intervalSeconds: 5,
      flowHandle: "fh-1",
    };
    expect(started()).toEqual(expected);
  });

  it("never links anywhere but github.com", () => {
    const state = started("https://evil.example/login/device");
    expect(state.kind === "awaiting" && state.verificationUri).toBe("https://github.com/login/device");
  });

  it("keeps waiting on pending and slows down by the server interval or by five seconds", () => {
    const pending = connectFlowReducer(started(), { type: "polled", response: { status: "pending" } });
    expect(pending).toMatchObject({ kind: "awaiting", pollCount: 1, intervalSeconds: 5 });

    const slowed = connectFlowReducer(pending, { type: "polled", response: { status: "slow_down", interval: 12 } });
    expect(slowed).toMatchObject({ kind: "awaiting", pollCount: 2, intervalSeconds: 12 });

    const slowedAgain = connectFlowReducer(slowed, { type: "polled", response: { status: "slow_down" } });
    expect(slowedAgain).toMatchObject({ kind: "awaiting", intervalSeconds: 17 });
  });

  it.each(["expired", "denied", "no_copilot_seat"] as const)("fails with %s", (status) => {
    expect(connectFlowReducer(started(), { type: "polled", response: { status } })).toEqual({
      kind: "failed",
      attempt: 0,
      reason: status,
    });
  });

  it("connects with the GitHub login", () => {
    expect(
      connectFlowReducer(started(), { type: "polled", response: { status: "connected", github_login: "octocat" } }),
    ).toEqual({ kind: "connected", attempt: 0, githubLogin: "octocat" });
  });

  it("expires locally once the device code lifetime has passed", () => {
    expect(connectFlowReducer(started(), { type: "timed_out" })).toEqual({
      kind: "failed",
      attempt: 0,
      reason: "expired",
    });
  });

  it("restarts with a new attempt after a failure", () => {
    const failed = connectFlowReducer(started(), { type: "errored", message: "boom" });
    const expected: ConnectFlowState = { kind: "failed", attempt: 0, reason: "error", message: "boom" };
    expect(failed).toEqual(expected);
    expect(connectFlowReducer(failed, { type: "restarted" })).toEqual({ kind: "starting", attempt: 1 });
  });

  it("ignores a late poll result after the flow left the awaiting state", () => {
    const connected = connectFlowReducer(started(), {
      type: "polled",
      response: { status: "connected", github_login: "octocat" },
    });
    expect(connectFlowReducer(connected, { type: "polled", response: { status: "expired" } })).toBe(connected);
    expect(connectFlowReducer(connected, { type: "errored", message: "late" })).toBe(connected);
  });
});
