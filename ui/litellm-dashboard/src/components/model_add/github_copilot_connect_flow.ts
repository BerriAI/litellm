import type { UserConnectionPollResponse, UserConnectionStartResponse } from "@/components/networking";

export const GITHUB_DEVICE_LOGIN_URL = "https://github.com/login/device";
const SLOW_DOWN_INCREMENT_SECONDS = 5;

export type ConnectFailureReason = "expired" | "denied" | "no_copilot_seat" | "error";

export type ConnectFlowState =
  | { readonly kind: "starting"; readonly attempt: number }
  | {
      readonly kind: "awaiting";
      readonly attempt: number;
      readonly pollCount: number;
      readonly userCode: string;
      readonly verificationUri: string;
      readonly expiresAt: number;
      readonly intervalSeconds: number;
      readonly flowHandle: string;
    }
  | { readonly kind: "connected"; readonly attempt: number; readonly githubLogin: string }
  | {
      readonly kind: "failed";
      readonly attempt: number;
      readonly reason: ConnectFailureReason;
      readonly message?: string;
    };

export type ConnectFlowEvent =
  | { readonly type: "started"; readonly response: UserConnectionStartResponse; readonly now: number }
  | { readonly type: "polled"; readonly response: UserConnectionPollResponse }
  | { readonly type: "timed_out" }
  | { readonly type: "errored"; readonly message: string }
  | { readonly type: "restarted" };

export const initialConnectFlowState: ConnectFlowState = { kind: "starting", attempt: 0 };

const safeVerificationUri = (uri: string): string =>
  uri.startsWith("https://github.com/") ? uri : GITHUB_DEVICE_LOGIN_URL;

const afterPoll = (
  state: Extract<ConnectFlowState, { kind: "awaiting" }>,
  response: UserConnectionPollResponse,
): ConnectFlowState => {
  switch (response.status) {
    case "pending":
      return { ...state, pollCount: state.pollCount + 1 };
    case "slow_down":
      return {
        ...state,
        pollCount: state.pollCount + 1,
        intervalSeconds: response.interval ?? state.intervalSeconds + SLOW_DOWN_INCREMENT_SECONDS,
      };
    case "connected":
      return { kind: "connected", attempt: state.attempt, githubLogin: response.github_login ?? "" };
    case "expired":
    case "denied":
    case "no_copilot_seat":
      return { kind: "failed", attempt: state.attempt, reason: response.status };
  }
};

export const connectFlowReducer = (state: ConnectFlowState, event: ConnectFlowEvent): ConnectFlowState => {
  switch (event.type) {
    case "restarted":
      return { kind: "starting", attempt: state.attempt + 1 };
    case "errored":
      return state.kind === "connected"
        ? state
        : { kind: "failed", attempt: state.attempt, reason: "error", message: event.message };
    case "started":
      return state.kind !== "starting"
        ? state
        : {
            kind: "awaiting",
            attempt: state.attempt,
            pollCount: 0,
            userCode: event.response.user_code,
            verificationUri: safeVerificationUri(event.response.verification_uri),
            expiresAt: event.now + event.response.expires_in * 1000,
            intervalSeconds: event.response.interval,
            flowHandle: event.response.flow_handle,
          };
    case "polled":
      return state.kind === "awaiting" ? afterPoll(state, event.response) : state;
    case "timed_out":
      return state.kind === "awaiting" ? { kind: "failed", attempt: state.attempt, reason: "expired" } : state;
  }
};

export const failureMessage = (state: Extract<ConnectFlowState, { kind: "failed" }>): string => {
  switch (state.reason) {
    case "expired":
      return "The device code expired before it was approved. Start again to get a new code.";
    case "denied":
      return "The request was denied on GitHub. Start again if this was a mistake.";
    case "no_copilot_seat":
      return "This GitHub account does not have GitHub Copilot access, so nothing was saved.";
    case "error":
      return state.message ?? "Something went wrong while connecting to GitHub.";
  }
};
