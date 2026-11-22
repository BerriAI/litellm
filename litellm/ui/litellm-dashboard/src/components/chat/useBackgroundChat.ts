import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { cancelBackgroundChat, isBackgroundRunPending, watchBackgroundChat } from "./backgroundChat";
import type { AssistantMessageUpdate, ChatBackgroundRun } from "./types";

export function useBackgroundChat(
  conversationId: string | null,
  run: ChatBackgroundRun | undefined,
  accessToken: string,
  updateMessage: (conversationId: string, update: AssistantMessageUpdate) => void,
) {
  const { t } = useTranslation();
  const [failure, setFailure] = useState<{ responseId: string; retry: number; message: string } | null>(null);
  const [retry, setRetry] = useState(0);
  const [cancellingId, setCancellingId] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const pending = isBackgroundRunPending(run);
  const responseId = run?.id;
  const cancelling = Boolean(responseId && cancellingId === responseId);
  const connectionError =
    pending && failure?.responseId === responseId && failure?.retry === retry ? failure.message : null;

  useEffect(() => {
    if (!conversationId || !responseId || !pending) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    const polling = {
      accessToken,
      responseId,
      signal: controller.signal,
      onUpdate: (update: AssistantMessageUpdate) => updateMessage(conversationId, update),
    };
    void watchBackgroundChat(polling, t).catch(() => {
      if (!controller.signal.aborted)
        setFailure({ responseId, retry, message: t("chat.useBackgroundChat.connectionLost") });
    });
    return () => {
      controller.abort();
      if (controllerRef.current === controller) controllerRef.current = null;
    };
  }, [conversationId, responseId, pending, accessToken, updateMessage, retry, t]);

  const reconnect = useCallback(() => setRetry((value) => value + 1), []);

  const cancel = useCallback(async () => {
    if (!conversationId || !responseId || cancelling) return;
    const controller = controllerRef.current;
    controller?.abort();
    setCancellingId(responseId);
    try {
      const update = await cancelBackgroundChat(accessToken, responseId, t);
      updateMessage(conversationId, update);
      if (controllerRef.current === controller) reconnect();
    } catch {
      if (controllerRef.current === controller) {
        setFailure({ responseId, retry, message: t("chat.useBackgroundChat.stopFailed") });
      }
    } finally {
      setCancellingId(null);
    }
  }, [conversationId, responseId, accessToken, cancelling, reconnect, retry, updateMessage, t]);

  return { pending, connectionError, cancelling, reconnect, cancel };
}
