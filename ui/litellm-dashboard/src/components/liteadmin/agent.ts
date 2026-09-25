import { z } from "zod";
import type { ChatMessage } from "@/components/chat/types";
import type { OperationContext, LiteAdminAction } from "./operations";

export const MAX_INPUT_LENGTH = 8_000;

export interface LiteAdminOptions extends OperationContext {
  model: string;
  messages: readonly Pick<ChatMessage, "role" | "content">[];
  managementBaseUrl: string;
  inferenceBaseUrl: string;
  onMessage: (text: string) => void;
}

const actionFields = {
  id: z.string(),
  name: z.string(),
  title: z.string(),
  arguments: z.record(z.unknown()),
  destructive: z.boolean(),
};
const actionSchema = z.object(actionFields);
const eventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("message"), text: z.string() }),
  z.object({ type: z.literal("approval"), action: actionSchema }),
  z.object({
    type: z.literal("result"),
    action: actionSchema,
    result: z.discriminatedUnion("status", [
      z.object({ status: z.literal("completed"), key: z.string().optional() }),
      z.object({ status: z.literal("unknown"), message: z.string() }),
    ]),
  }),
  z.object({ type: z.literal("error"), message: z.string() }),
  z.object({ type: z.literal("done") }),
]);
type InferenceTarget =
  | { baseUrl: string; requiresConsent: boolean; error: null }
  | { baseUrl: null; requiresConsent: false; error: string };

export function resolveInferenceTarget(candidate: string, managementBaseUrl: string, pageUrl: string): InferenceTarget {
  const invalid: InferenceTarget = {
    baseUrl: null,
    requiresConsent: false,
    error: "The gateway must be a valid HTTP(S) URL without credentials, a query, or a fragment.",
  };
  try {
    const page = new URL(pageUrl);
    const management = new URL(managementBaseUrl.trim() || page.origin, `${page.origin}/`);
    const target = new URL(candidate.trim() || management.href, `${page.origin}/`);
    if (![page, management, target].every((url) => ["http:", "https:"].includes(url.protocol))) return invalid;
    if ([management, target].some((url) => url.username || url.password || /[?#]/.test(url.href))) return invalid;
    if (target.protocol === "http:" && (page.protocol === "https:" || management.protocol === "https:")) {
      return {
        baseUrl: null,
        requiresConsent: false,
        error: "Inference must use HTTPS when the dashboard or management gateway uses HTTPS.",
      };
    }
    return { baseUrl: target.href, requiresConsent: target.origin !== management.origin, error: null };
  } catch {
    return invalid;
  }
}

export type AgentSocket = Pick<WebSocket, "onopen" | "onmessage" | "onerror" | "onclose" | "send" | "close">;

export async function runLiteAdmin(
  options: LiteAdminOptions,
  connect: (url: string) => AgentSocket = (url) => new WebSocket(url),
): Promise<void> {
  const active = () => {
    options.signal.throwIfAborted();
    options.assertCurrent();
  };
  active();
  const messages = options.messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      role: message.role,
      content: message.role === "assistant" ? message.content.slice(0, MAX_INPUT_LENGTH) : message.content,
    }))
    .slice(-20);
  if (messages.some((message) => message.content.length > MAX_INPUT_LENGTH)) {
    throw new Error("Keep each message under 8,000 characters, or start a new chat.");
  }
  const url = new URL(`${options.managementBaseUrl.replace(/\/$/, "")}/liteadmin/chat`, window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  const socket = connect(url.href);
  await new Promise<void>((resolve, reject) => {
    let finished = false;
    let submitted = false;
    let pending: LiteAdminAction | null = null;
    const finish = (error?: unknown) => {
      if (finished) return;
      finished = true;
      options.signal.removeEventListener("abort", abort);
      const uncertain = error && submitted && !options.signal.aborted;
      if (uncertain && pending) {
        try {
          options.onResult(pending, {
            status: "unknown",
            message: "The change could not be verified. Check the resource before trying again.",
          });
        } catch {}
      }
      socket.close();
      if (error) reject(error);
      else resolve();
    };
    const abort = () => finish(new DOMException("The request was cancelled", "AbortError"));
    options.signal.addEventListener("abort", abort, { once: true });
    socket.onopen = () => {
      try {
        active();
        socket.send(
          JSON.stringify({
            access_token: options.accessToken,
            chat: {
              model: options.model,
              messages,
              inference_base_url: options.inferenceBaseUrl,
            },
          }),
        );
      } catch (error) {
        finish(error);
      }
    };
    socket.onmessage = async ({ data }) => {
      if (finished) return;
      try {
        active();
        const event = eventSchema.parse(JSON.parse(String(data)));
        switch (event.type) {
          case "message":
            options.onMessage(event.text);
            break;
          case "approval": {
            if (pending) throw new Error("LiteAdmin received overlapping approvals.");
            pending = event.action;
            const approved = await options.confirm(event.action);
            active();
            if (finished) return;
            submitted = approved;
            socket.send(JSON.stringify({ id: event.action.id, approved }));
            if (!approved) finish();
            break;
          }
          case "result":
            if (!pending || event.action.id !== pending.id || !submitted)
              throw new Error("LiteAdmin received an unexpected action result.");
            options.onResult(event.action, event.result);
            pending = null;
            submitted = false;
            break;
          case "error":
            finish(new Error(event.message));
            break;
          case "done":
            if (pending) throw new Error("LiteAdmin ended before confirming the action outcome.");
            finish();
            break;
        }
      } catch (error) {
        finish(error);
      }
    };
    socket.onerror = () =>
      finish(new Error("Could not connect to LiteAdmin. Check that the gateway supports WebSocket connections."));
    socket.onclose = () => finish(new Error("The LiteAdmin connection closed before the request completed."));
  });
}
