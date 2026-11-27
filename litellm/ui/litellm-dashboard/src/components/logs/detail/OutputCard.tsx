/**
 * OutputCard - Displays output message with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { copyToClipboard } from "@/utils/dataUtils";
import { ParsedMessage } from "./prettyMessagesTypes";
import { SectionHeader } from "./SectionHeader";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

const assistantRoleLabel = "ASSISTANT";

interface OutputCardProps {
  message: ParsedMessage | null;
  completionTokens?: number;
  outputCost?: number;
}

export function OutputCard({ message, completionTokens, outputCost }: OutputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const { t } = useTranslation();

  const handleCopy = () => {
    if (!message) return;

    void copyToClipboard(message.content || "", t("logs.detail.outputCard.copied"));
  };

  return (
    <div className="overflow-hidden rounded-md border border-border">
      <SectionHeader
        type="output"
        tokens={completionTokens}
        cost={outputCost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      <div
        className="overflow-hidden transition-[max-height,opacity] duration-300 ease-out"
        style={{ maxHeight: isCollapsed ? "0px" : "10000px", opacity: isCollapsed ? 0 : 1 }}
      >
        <div className="px-4 py-3">
          {message ? (
            <SimpleMessageBlock label={assistantRoleLabel} content={message.content} toolCalls={message.toolCalls} />
          ) : (
            <span className="text-[13px] text-muted-foreground italic">{t("logs.detail.outputCard.noResponse")}</span>
          )}
        </div>
      </div>
    </div>
  );
}
