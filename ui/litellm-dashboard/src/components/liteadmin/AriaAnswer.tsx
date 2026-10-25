import { useLanguage } from "@/contexts/LanguageContext";
import { useRef } from "react";
import { Download } from "lucide-react";
import { ChatMessageContent } from "@/components/chat/ChatMessages";
import type { ChatMessage } from "@/components/chat/types";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { ARIA_AGENT_LABEL } from "./aria-brand";
import { exportAriaAnswer } from "./answerExport";
import { UsageAnswerChart } from "./UsageAnswerChart";
import type { UsageChart } from "./usageChart";

export function AriaAnswer({
  message,
  question,
  chart,
}: {
  message: ChatMessage;
  question?: string;
  chart?: UsageChart;
}) {
  const { t } = useLanguage();
  const answerRef = useRef<HTMLDivElement>(null);
  const exportAnswer = () => {
    if (!answerRef.current || !exportAriaAnswer(question ?? "", answerRef.current)) {
      toast.error(t("Could not open the PDF print dialog. Please try again."));
    }
  };
  return (
    <article
      aria-label={t("{ARIA_AGENT_LABEL} answer", { ARIA_AGENT_LABEL: String(ARIA_AGENT_LABEL) })}
      className="space-y-3"
    >
      <p className="text-xs font-medium text-muted-foreground">{ARIA_AGENT_LABEL}</p>
      <div ref={answerRef} className="space-y-4">
        {chart && <UsageAnswerChart chart={chart} />}
        <ChatMessageContent
          message={message}
          allowImages={false}
          isLastMessage={false}
          isStreaming={false}
          actions={
            <Button variant="ghost" size="sm" onClick={exportAnswer} data-export-exclude>
              <Download className="size-3.5" />
              {t("Export answer")}
            </Button>
          }
        />
      </div>
    </article>
  );
}
