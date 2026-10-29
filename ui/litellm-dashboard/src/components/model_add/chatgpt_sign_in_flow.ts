import type {
  ChatGPTLoginPollResponse,
  ChatGPTLoginStartRequest,
  ChatGPTLoginStartResponse,
  CredentialItem,
} from "@/components/networking";

export const CHATGPT_DEVICE_LOGIN_URL = "https://auth.openai.com/codex/device";
const CHATGPT_DEVICE_LOGIN_HOST = "auth.openai.com";

export type ChatGPTOwnerKind = "team" | "key";

export const CHATGPT_OWNER_FIELDS: Readonly<Record<ChatGPTOwnerKind, string>> = {
  team: "chatgpt_owner_team_id",
  key: "chatgpt_owner_key_hash",
};

export type SignInFailureReason = "expired" | "error";

export type SignInFlowState =
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
  | { readonly kind: "connected"; readonly attempt: number }
  | {
      readonly kind: "failed";
      readonly attempt: number;
      readonly reason: SignInFailureReason;
      readonly message?: string;
    };

export type SignInFlowEvent =
  | { readonly type: "started"; readonly response: ChatGPTLoginStartResponse; readonly now: number }
  | { readonly type: "polled"; readonly response: ChatGPTLoginPollResponse }
  | { readonly type: "timed_out" }
  | { readonly type: "errored"; readonly message: string }
  | { readonly type: "restarted" };

export const initialSignInFlowState: SignInFlowState = { kind: "starting", attempt: 0 };

export const chatGPTStartRequest = (
  ownerKind: ChatGPTOwnerKind,
  ownerId: string,
  displayName: string,
): ChatGPTLoginStartRequest => ({
  ownerTeamId: ownerKind === "team" ? ownerId : null,
  ownerKeyHash: ownerKind === "key" ? ownerId : null,
  displayName: displayName.trim() === "" ? null : displayName.trim(),
});

export interface StoredChatGPTOwner {
  readonly teamId: string | null;
  readonly keyHash: string | null;
  readonly kind: ChatGPTOwnerKind;
}

export const storedChatGPTOwner = (credential: CredentialItem | null): StoredChatGPTOwner => {
  const teamId = credential?.credential_info.chatgpt_owner_team_id ?? null;
  const keyHash = credential?.credential_info.chatgpt_owner_key_hash ?? null;
  return { teamId, keyHash, kind: keyHash ? "key" : "team" };
};

export interface ChatGPTSignIn {
  readonly credentialName: string;
  readonly request: ChatGPTLoginStartRequest;
}

export const chatGPTSignInFrom = (
  ownerKind: ChatGPTOwnerKind,
  values: Readonly<Record<string, unknown>>,
): ChatGPTSignIn => {
  const ownerId = values[CHATGPT_OWNER_FIELDS[ownerKind]];
  const displayName = values.display_name;
  return {
    credentialName: String(values.credential_name),
    request: chatGPTStartRequest(
      ownerKind,
      typeof ownerId === "string" ? ownerId.trim() : "",
      typeof displayName === "string" ? displayName : "",
    ),
  };
};

const SUBMIT_LABELS = {
  chatgpt: { add: "Sign in with ChatGPT", edit: "Sign in again" },
  credential: { add: "Add Credential", edit: "Update Credential" },
} as const;

export const credentialSubmitLabel = (signsInWithChatGPT: boolean, mode: "add" | "edit"): string =>
  SUBMIT_LABELS[signsInWithChatGPT ? "chatgpt" : "credential"][mode];

const safeVerificationUri = (uri: string): string => {
  try {
    const url = new URL(uri);
    const hasCredentials = url.username !== "" || url.password !== "";
    const isDefaultHttps = url.protocol === "https:" && url.port === "";
    const isSafeUrl = isDefaultHttps && !hasCredentials && url.hostname === CHATGPT_DEVICE_LOGIN_HOST;
    return isSafeUrl ? uri : CHATGPT_DEVICE_LOGIN_URL;
  } catch {
    return CHATGPT_DEVICE_LOGIN_URL;
  }
};

const afterPoll = (
  state: Extract<SignInFlowState, { kind: "awaiting" }>,
  response: ChatGPTLoginPollResponse,
): SignInFlowState => {
  switch (response.status) {
    case "pending":
      return { ...state, pollCount: state.pollCount + 1 };
    case "connected":
      return { kind: "connected", attempt: state.attempt };
  }
};

export const signInFlowReducer = (state: SignInFlowState, event: SignInFlowEvent): SignInFlowState => {
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

export const signInFailureMessage = (state: Extract<SignInFlowState, { kind: "failed" }>): string => {
  switch (state.reason) {
    case "expired":
      return "The sign-in code expired before it was used. Start again to get a new code.";
    case "error":
      return state.message ?? "Something went wrong while signing in to ChatGPT.";
  }
};
