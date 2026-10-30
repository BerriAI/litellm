import type { ReactNode } from "react";
import { ArrowUpRight, Bug, Clock3, Code2, ListChecks } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Conversation } from "../types";
import "./moyai-chat.css";

const starters = [
  {
    label: "Review code",
    detail: "Find the edge cases",
    prompt: "Review my changes for bugs and improvements.",
    Icon: Code2,
  },
  { label: "Fix a bug", detail: "Trace it to the source", prompt: "Help me investigate and fix this bug: ", Icon: Bug },
  {
    label: "Plan a feature",
    detail: "Make the next change clear",
    prompt: "Help me plan and implement this feature: ",
    Icon: ListChecks,
  },
];

export default function MoyaiChatHome({
  children,
  imageMode,
  draft,
  busy,
  conversations,
  onSelectConversation,
  onPrompt,
}: {
  children: ReactNode;
  imageMode: boolean;
  draft: string;
  busy: boolean;
  conversations: Conversation[];
  onSelectConversation: (id: string) => void;
  onPrompt: (prompt: string) => void;
}) {
  const recent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 3);
  return (
    <section className="moyai-chat-home">
      <div className="moyai-chat-space-backdrop" aria-hidden="true" />
      <div className="moyai-chat-home-content">
        <header className="moyai-chat-home-header">
          <h1>
            {imageMode ? (
              "What are we creating?"
            ) : (
              <>
                What are we <span>building?</span>
              </>
            )}
          </h1>
        </header>
        {children}
        <section className="moyai-chat-home-recent" aria-labelledby="recent-chats-heading">
          <div className="moyai-chat-home-recent-header">
            <h2 id="recent-chats-heading">Recent chats</h2>
          </div>
          {recent.length === 0 && (
            <p className="text-muted-foreground" role="status">
              No recent chats in this browser yet.
            </p>
          )}
          <ul>
            {recent.map((conversation) => (
              <li key={conversation.id}>
                <Button
                  variant="ghost"
                  className="moyai-chat-home-recent-link"
                  onClick={() => onSelectConversation(conversation.id)}
                >
                  <Clock3 className="size-3.5" />
                  <span>{conversation.title}</span>
                  <ArrowUpRight className="size-3.5" />
                </Button>
              </li>
            ))}
          </ul>
        </section>
        {!imageMode && (
          <div
            className="moyai-chat-home-starters"
            role="group"
            aria-label="Start with a prompt"
            aria-hidden={Boolean(draft.trim())}
          >
            <p className="moyai-chat-home-starters-heading">Start with a prompt</p>
            {starters.map(({ label, detail, prompt, Icon }) => (
              <Button
                key={label}
                variant="ghost"
                disabled={busy || Boolean(draft.trim())}
                onClick={() => onPrompt(prompt)}
              >
                <Icon />
                <span className="moyai-chat-home-starter-label">{label}</span>
                <span className="moyai-chat-home-starter-detail">{detail}</span>
                <ArrowUpRight className="moyai-chat-home-starter-arrow" />
              </Button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
