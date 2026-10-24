import { v4 as uuidv4, v5 as uuidv5 } from "uuid";
import { serverRootPath } from "@/lib/serverRootPath";
import { captureChatRequest } from "@/lib/http/chatTelemetry";

// Called only by Chat's accepted-submit handler. Daily unique users = DAU;
// event count = total requests. Page visits never count as activity.
export async function recordChatQuery(userId: string): Promise<void> {
  // Match docs: development previews never contribute to production metrics.
  if (process.env.NODE_ENV !== "production" || typeof window === "undefined" || !userId) return;
  try {
    // Stable across browsers for an account on this deployment. The payload
    // excludes its raw account ID, email, and gateway origin.
    const root = serverRootPath.replace(/^\/+|\/+$/g, "");
    const identity = JSON.stringify([window.location.origin, root, userId]);
    const distinctId = uuidv5(identity, uuidv5.URL);
    await captureChatRequest(distinctId, uuidv4());
  } catch {
    // Analytics failures must never interrupt a request or trigger a retry.
  }
}
