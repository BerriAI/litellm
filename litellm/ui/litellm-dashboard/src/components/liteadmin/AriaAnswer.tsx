import { useRef } from "react";
import { Download } from "lucide-react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  const answerRef = useRef<HTMLDivElement>(null);
  const exportAnswer = () => {
    if (!answerRef.current || !exportAriaAnswer(question ?? "", answerRef.current)) {
      toast.error(t("aria.answer.exportFailed"));
    }
  };
  return (
    <article aria-label={t("aria.answer.ariaLabel", { agent: ARIA_AGENT_LABEL })} className="space-y-3">
      <div ref={answerRef} className="space-y-4">
        {chart && <UsageAnswerChart chart={chart} />}
        <ChatMessageContent
          message={message}
          allowImages={false}
          isLastMessage={false}
          isStreaming={false}
          actions={
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={exportAnswer}
              data-export-exclude
              aria-label={t("aria.answer.export")}
              title={t("aria.answer.export")}
              className="text-muted-foreground"
            >
              <Download className="size-3.5" />
            </Button>
          }
        />
      </div>
    </article>
  );
}
