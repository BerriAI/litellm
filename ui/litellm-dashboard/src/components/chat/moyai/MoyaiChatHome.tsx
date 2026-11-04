import type { ReactNode } from "react";
import { Bug, Code2, ListChecks } from "lucide-react";
import { Button } from "@/components/ui/button";
import "./moyai-chat.css";

const starters = [
  {
    label: "Review code",
    prompt: "Review my changes for bugs and improvements.",
    Icon: Code2,
  },
  { label: "Fix a bug", prompt: "Help me investigate and fix this bug: ", Icon: Bug },
  {
    label: "Plan a feature",
    prompt: "Help me plan and implement this feature: ",
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
  return (
    <section className="moyai-chat-home">
      <div className="moyai-chat-home-content">
        <header className="moyai-chat-home-header">
          <h1>
            {imageMode && "What are we creating?"}
            {!imageMode && researchMode && (
              <>
                What should we <span>research?</span>
              </>
            )}
            {!imageMode && !researchMode && (
              <>
                What are we <span>working on?</span>
              </>
            )}
          </h1>
        </header>
        {children}
        {!imageMode && (
          <div
            className="moyai-chat-home-starters"
            role="group"
            aria-label="Start with a prompt"
            aria-hidden={Boolean(draft.trim())}
          >
            {starters.map(({ label, prompt, Icon }) => (
              <Button
                key={label}
                variant="outline"
                className="dark:bg-transparent"
                disabled={busy || Boolean(draft.trim())}
                onClick={() => onPrompt(prompt)}
              >
                <Icon />
                <span className="moyai-chat-home-starter-label">{label}</span>
              </Button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
