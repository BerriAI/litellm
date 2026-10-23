import { v4 as uuidv4 } from "uuid";
import { queryClient } from "@/contexts/ReactQueryProvider";
import { backgroundFetchClient } from "@/lib/http/api";
import { captureChatEvents, type ChatPosthogEvent } from "@/lib/http/chatPosthog";
import { TELEMETRY_WINDOW_MS } from "./uiEvents";

const chatPosthogQuery = {
  queryKey: ["telemetry", "chatPosthogEnabled"],
  queryFn: async (): Promise<boolean> => {
    const { data } = await backgroundFetchClient.GET("/telemetry/ui_events/enabled");
    return data?.chat_posthog_enabled === true;
  },
  staleTime: TELEMETRY_WINDOW_MS,
  retry: false,
};

const activeDays = new Set<string>();

async function recordChatActivity(userId: string, query: boolean): Promise<void> {
  // Match docs: development previews never contribute to production metrics.
  if (process.env.NODE_ENV !== "production" || typeof window === "undefined" || !userId) return;
  const timestamp = new Date().toISOString();
  let activeKey: string | undefined;
  try {
    if (!(await queryClient.fetchQuery(chatPosthogQuery))) return;
    // Only an opaque ID leaves the browser. Account IDs just scope local storage.
    const storageKey = `litellm-chat-telemetry:${userId}`;
    let distinctId = localStorage.getItem(storageKey);
    if (!distinctId) {
      distinctId = uuidv4();
      localStorage.setItem(storageKey, distinctId);
    }
    const dayKey = `${distinctId}:${timestamp.slice(0, 10)}`;
    const events: ChatPosthogEvent[] = [];
    if (!activeDays.has(dayKey)) {
      activeKey = dayKey;
      activeDays.add(dayKey);
      events.push("LiteLLM Chat UI");
    }
    if (query) events.push("LiteLLM Chat UI Query");
    if (events.length > 0) await captureChatEvents(distinctId, events, timestamp);
  } catch {
    // Telemetry must never interrupt Chat; retry a failed daily marker on next activity.
    if (activeKey) activeDays.delete(activeKey);
  }
}

// DAU = daily unique distinct IDs for this event; repeat visits do not inflate DAU.
export const recordChatActive = (userId: string): Promise<void> => recordChatActivity(userId, false);

// Total queries = total events, including explicit resends and provider failures.
export const recordChatQuery = (userId: string): Promise<void> => recordChatActivity(userId, true);
