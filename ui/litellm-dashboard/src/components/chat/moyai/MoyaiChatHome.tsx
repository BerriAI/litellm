import type { ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { ArrowUpRight, Bug, Clock3, Code2, ListChecks, type LucideIcon } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import type { Conversation } from "../types";
import "./moyai-chat.css";

const starters: { label: ParseKeys; detail: ParseKeys; prompt: ParseKeys; Icon: LucideIcon }[] = [
  {
    label: "chat.moyaiHome.starters.review.label",
    detail: "chat.moyaiHome.starters.review.detail",
    prompt: "chat.moyaiHome.starters.review.prompt",
    Icon: Code2,
  },
  {
    label: "chat.moyaiHome.starters.fixBug.label",
    detail: "chat.moyaiHome.starters.fixBug.detail",
    prompt: "chat.moyaiHome.starters.fixBug.prompt",
    Icon: Bug,
  },
  {
    label: "chat.moyaiHome.starters.planFeature.label",
    detail: "chat.moyaiHome.starters.planFeature.detail",
    prompt: "chat.moyaiHome.starters.planFeature.prompt",
    Icon: ListChecks,
  },
];

export default function MoyaiChatHome({
  children,
  imageMode,
  researchMode,
  draft,
  busy,
  conversations,
  onSelectConversation,
  onPrompt,
}: {
  children: ReactNode;
  imageMode: boolean;
  researchMode: boolean;
  draft: string;
  busy: boolean;
  conversations: Conversation[];
  onSelectConversation: (id: string) => void;
  onPrompt: (prompt: string) => void;
}) {
  const { t } = useTranslation();
  const recent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 3);
  return (
    <section className="moyai-chat-home">
      <div className="moyai-chat-space-backdrop" aria-hidden="true" />
      <div className="moyai-chat-home-content">
        <header className="moyai-chat-home-header">
          <h1>
            {imageMode && t("chat.moyaiHome.headingImage")}
            {!imageMode && researchMode && (
              <>
                <Trans i18nKey="chat.moyaiHome.headingResearch" components={{ highlight: <span /> }} />
              </>
            )}
            {!imageMode && !researchMode && (
              <>
                <Trans i18nKey="chat.moyaiHome.headingChat" components={{ highlight: <span /> }} />
              </>
            )}
          </h1>
        </header>
        {children}
        <section className="moyai-chat-home-recent" aria-labelledby="recent-chats-heading">
          <div className="moyai-chat-home-recent-header">
            <h2 id="recent-chats-heading">{t("chat.moyaiHome.recentChats")}</h2>
          </div>
          {recent.length === 0 && (
            <p className="text-muted-foreground" role="status">
              {t("chat.moyaiHome.noRecentChats")}
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
            aria-label={t("chat.moyaiHome.startersLabel")}
            aria-hidden={Boolean(draft.trim())}
          >
            <p className="moyai-chat-home-starters-heading">{t("chat.moyaiHome.startersHeading")}</p>
            {starters.map(({ label, detail, prompt, Icon }) => (
              <Button
                key={label}
                variant="ghost"
                disabled={busy || Boolean(draft.trim())}
                onClick={() => onPrompt(t(prompt))}
              >
                <Icon />
                <span className="moyai-chat-home-starter-label">{t(label)}</span>
                <span className="moyai-chat-home-starter-detail">{t(detail)}</span>
                <ArrowUpRight className="moyai-chat-home-starter-arrow" />
              </Button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
