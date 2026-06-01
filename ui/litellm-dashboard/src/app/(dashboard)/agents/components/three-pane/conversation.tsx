"use client";

/**
 * Conversation — middle pane of the three-pane Session view.
 *
 * Renders the union of:
 *   - the initial conversation snapshot (from /v2/sessions/{sid}/conversation)
 *   - live events from useSessionEventStream (user_message, assistant_message,
 *     tool_call, file_diff)
 *
 * file_diff events fold into a single "N Files Changed" accordion at the
 * bottom (cumulative across the run, per the LIT-2881 spec).
 */
import { useMemo } from "react";
import { Empty, Typography } from "antd";
import MessageBubble from "@/app/(dashboard)/agents/components/three-pane/message-bubble";
import ToolCallCard from "@/app/(dashboard)/agents/components/three-pane/tool-call-card";
import FilesChangedAccordion from "@/app/(dashboard)/agents/components/three-pane/files-changed-accordion";
import Composer from "@/app/(dashboard)/agents/components/three-pane/composer";
import type {
  CloudAgentConversationMessage,
  CloudAgentRunEvent,
  FileDiffPayload,
  ToolCallPayload,
} from "@/types/cloud-agents";

const { Title } = Typography;

interface ConversationProps {
  sessionId: string;
  accessToken: string | null;
  initialMessages: CloudAgentConversationMessage[];
  events: CloudAgentRunEvent[];
}

interface DisplayItem {
  key: string;
  kind: "message" | "tool_call";
  message?: CloudAgentConversationMessage;
  toolCall?: ToolCallPayload;
}

function buildDisplayItems(
  initialMessages: CloudAgentConversationMessage[],
  events: CloudAgentRunEvent[],
): { items: DisplayItem[]; diffs: FileDiffPayload[] } {
  const items: DisplayItem[] = initialMessages.map((m) => ({
    key: `msg-${m.id}`,
    kind: "message",
    message: m,
  }));
  const diffs: FileDiffPayload[] = [];
  for (const evt of events) {
    if (evt.type === "user_message" || evt.type === "assistant_message") {
      const role = evt.type === "user_message" ? "user" : "assistant";
      const content = String((evt.payload as { content?: unknown }).content ?? "");
      items.push({
        key: `evt-${evt.seq}`,
        kind: "message",
        message: {
          id: `evt-${evt.seq}`,
          role: role as CloudAgentConversationMessage["role"],
          content,
          created_at: evt.created_at,
        },
      });
    } else if (evt.type === "tool_call") {
      items.push({
        key: `evt-${evt.seq}`,
        kind: "tool_call",
        toolCall: evt.payload as unknown as ToolCallPayload,
      });
    } else if (evt.type === "file_diff") {
      diffs.push(evt.payload as unknown as FileDiffPayload);
    }
  }
  return { items, diffs };
}

export default function Conversation({ sessionId, accessToken, initialMessages, events }: ConversationProps) {
  const { items, diffs } = useMemo(() => buildDisplayItems(initialMessages, events), [initialMessages, events]);

  return (
    <div className="flex h-full flex-col" data-testid="conversation-pane">
      <div className="border-b border-gray-200 px-4 py-2">
        <Title level={5} className="!mb-0">
          Conversation
        </Title>
      </div>
      <div className="flex-1 overflow-y-auto p-4" data-testid="conversation-scroll">
        {items.length === 0 && diffs.length === 0 ? (
          <Empty description="No messages yet" />
        ) : (
          <>
            {items.map((it) =>
              it.kind === "message" && it.message ? (
                <MessageBubble
                  key={it.key}
                  role={it.message.role}
                  content={it.message.content}
                  timestamp={it.message.created_at}
                />
              ) : it.kind === "tool_call" && it.toolCall ? (
                <ToolCallCard key={it.key} tool={it.toolCall.tool} input={it.toolCall.input} />
              ) : null,
            )}
            <FilesChangedAccordion diffs={diffs} />
          </>
        )}
      </div>
      <Composer sessionId={sessionId} accessToken={accessToken} />
    </div>
  );
}
