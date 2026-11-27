import type { TFunction } from "i18next";
import type { Conversation } from "./types";

export const DEFAULT_CONVERSATION_TITLE = "New conversation";

export function isDefaultConversationTitle(conversation: Conversation): boolean {
  return (
    conversation.title === DEFAULT_CONVERSATION_TITLE &&
    !conversation.hasCustomTitle &&
    !conversation.messages.some((message) => message.role === "user")
  );
}

export function displayConversationTitle(conversation: Conversation, t: TFunction): string {
  return isDefaultConversationTitle(conversation) ? t("chat.chatHistory.newConversation") : conversation.title;
}
