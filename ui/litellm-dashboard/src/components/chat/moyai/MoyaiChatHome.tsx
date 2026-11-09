import type { ReactNode } from "react";
import type { ParseKeys } from "i18next";
import { Bug, Code2, ListChecks, type LucideIcon } from "lucide-react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import "./moyai-chat.css";

const starters: { label: ParseKeys; prompt: ParseKeys; Icon: LucideIcon }[] = [
  {
    label: "chat.moyaiHome.starters.review.label",
    prompt: "chat.moyaiHome.starters.review.prompt",
    Icon: Code2,
  },
  {
    label: "chat.moyaiHome.starters.fixBug.label",
    prompt: "chat.moyaiHome.starters.fixBug.prompt",
    Icon: Bug,
  },
  {
    label: "chat.moyaiHome.starters.planFeature.label",
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
  onPrompt,
}: {
  children: ReactNode;
  imageMode: boolean;
  researchMode: boolean;
  draft: string;
  busy: boolean;
  onPrompt: (prompt: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <section className="moyai-chat-home">
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
        {!imageMode && (
          <div
            className="moyai-chat-home-starters"
            role="group"
            aria-label={t("chat.moyaiHome.startersLabel")}
            aria-hidden={Boolean(draft.trim())}
          >
            {starters.map(({ label, prompt, Icon }) => (
              <Button
                key={label}
                variant="outline"
                className="dark:bg-transparent"
                disabled={busy || Boolean(draft.trim())}
                onClick={() => onPrompt(t(prompt))}
              >
                <Icon />
                <span className="moyai-chat-home-starter-label">{t(label)}</span>
              </Button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
