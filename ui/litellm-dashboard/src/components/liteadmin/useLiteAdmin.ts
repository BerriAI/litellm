import { useLayoutEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { ChatMessage } from "@/components/chat/types";
import { getProxyBaseUrl } from "@/components/networking";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { toast } from "@/lib/toast";
import { getCookie } from "@/utils/cookieUtils";
import { checkTokenValidity } from "@/utils/jwtUtils";
import { runLiteAdmin } from "./agent";
import type { ActionResult, LiteAdminAction } from "./operations";

type ActionState = { status: "review" | "applying" | "cancelled" } | ActionResult;
export type ActionEntry = { kind: "action"; id: string; action: LiteAdminAction } & ActionState;
export type ConversationEntry =
  | { kind: "message"; id: string; message: ChatMessage }
  | ActionEntry
  | { kind: "error"; id: string; text: string };
export interface LiteAdminSession {
  token: string;
  accessToken: string;
  managementBaseUrl: string;
  inferenceBaseUrl: string;
}

type Phase = "idle" | "thinking" | "review" | "applying";
type ActiveTurn = {
  controller: AbortController;
  outcome: "none" | "submitted" | "completed" | "unknown";
  approval?: { id: string; resolve: (approved: boolean) => void };
};
const INTERRUPTED_WRITE = "A submitted change may have completed. Check the relevant page before trying again.";
const RESOURCE_QUERIES = new Set([
  "keys",
  "infiniteKeys",
  "deletedKeys",
  "infiniteKeyAliases",
  "teams",
  "teamsTable",
  "infiniteTeams",
  "deletedTeams",
  "users",
  "infiniteUsers",
  "userLookup",
  "userList",
  "budgets",
]);

export function useLiteAdmin(session: LiteAdminSession) {
  const queryClient = useQueryClient();
  const [entries, setEntries] = useState<ConversationEntry[]>([]);
  const [phase, setPhase] = useState<Phase>("idle");
  const active = useRef<ActiveTurn | null>(null);

  useLayoutEffect(
    () => () => {
      const turn = active.current;
      active.current = null;
      if (turn?.outcome === "submitted") toast.warning(INTERRUPTED_WRITE);
      turn?.controller.abort();
      turn?.approval?.resolve(false);
    },
    [],
  );

  const updateAction = (id: string, result: ActionState) => {
    setEntries((current) =>
      current.map((entry) =>
        entry.kind === "action" && entry.id === id ? { kind: "action", id, action: entry.action, ...result } : entry,
      ),
    );
  };

  const stop = () => {
    const turn = active.current;
    if (turn?.outcome === "submitted") return false;
    active.current = null;
    turn?.controller.abort();
    if (turn?.approval) {
      updateAction(turn.approval.id, { status: "cancelled" });
      turn.approval.resolve(false);
    }
    setPhase("idle");
    return true;
  };

  const answer = (id: string, approved: boolean) => {
    const turn = active.current;
    if (!turn?.approval || turn.approval.id !== id) return;
    const { resolve } = turn.approval;
    turn.approval = undefined;
    if (approved) {
      turn.outcome = "submitted";
      updateAction(id, { status: "applying" });
      setPhase("applying");
    } else {
      updateAction(id, { status: "cancelled" });
      stop();
    }
    resolve(approved);
  };

  const send = async (text: string, model: string) => {
    if (active.current || !text.trim()) return;
    const turn: ActiveTurn = { controller: new AbortController(), outcome: "none" };
    active.current = turn;
    const message: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text.trim(), timestamp: Date.now() };
    const history = entries.flatMap<Pick<ChatMessage, "role" | "content">>((entry) => {
      if (entry.kind === "message") return [entry.message];
      if (entry.kind !== "action" || entry.status === "review" || entry.status === "applying") return [];
      const receipt = { operation: entry.action.name, status: entry.status, arguments: entry.action.arguments };
      return [{ role: "assistant", content: `Gateway action receipt: ${JSON.stringify(receipt)}` }];
    });
    setEntries((current) => [...current, { kind: "message", id: message.id, message }]);
    setPhase("thinking");
    const sameSession = () =>
      getCookie("token") === session.token &&
      checkTokenValidity(session.token) &&
      getProxyBaseUrl() === session.managementBaseUrl;
    const assertCurrent = () => {
      turn.controller.signal.throwIfAborted();
      if (active.current !== turn || !sameSession()) throw new DOMException("Session changed", "AbortError");
    };
    try {
      const options: Parameters<typeof runLiteAdmin>[0] = {
        ...session,
        model,
        messages: [...history, message],
        signal: turn.controller.signal,
        assertCurrent,
        onMessage: (content) => {
          assertCurrent();
          const reply: ChatMessage = { id: crypto.randomUUID(), role: "assistant", content, timestamp: Date.now() };
          setEntries((current) => [...current, { kind: "message", id: reply.id, message: reply }]);
        },
        confirm: (action) =>
          new Promise<boolean>((resolve) => {
            assertCurrent();
            turn.approval = { id: action.id, resolve };
            setEntries((current) => [...current, { kind: "action", id: action.id, action, status: "review" }]);
            setPhase("review");
          }),
        onResult: (action, result) => {
          assertCurrent();
          turn.outcome = result.status;
          updateAction(action.id, result);
          setPhase("thinking");
          if (result.status === "completed") {
            void queryClient.invalidateQueries({
              predicate: (query) => RESOURCE_QUERIES.has(String(query.queryKey[0])),
            });
          }
        },
      };
      await runLiteAdmin(options);
    } catch (error) {
      if (active.current !== turn) return;
      if (!sameSession()) {
        if (turn.outcome === "submitted") toast.warning(INTERRUPTED_WRITE);
        setEntries([]);
      } else if (!turn.controller.signal.aborted && turn.outcome !== "unknown") {
        const context = turn.outcome === "completed" ? "Completed changes are shown above. " : "";
        setEntries((current) => [
          ...current,
          { kind: "error", id: crypto.randomUUID(), text: context + extractProxyErrorMessage(error) },
        ]);
      }
    } finally {
      if (active.current === turn) {
        active.current = null;
        if (turn.approval) updateAction(turn.approval.id, { status: "cancelled" });
        turn.approval?.resolve(false);
        setPhase("idle");
      }
    }
  };

  return {
    entries,
    phase,
    send,
    answer,
    stop,
    reset: () => {
      if (stop()) setEntries([]);
    },
  };
}
