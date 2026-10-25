/**
 * HistoryTree - Collapsible tree view for message history
 * Shows arrow indicator and message count
 */

import { useLanguage } from "@/contexts/LanguageContext";
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ParsedMessage } from "./prettyMessagesTypes";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

interface HistoryTreeProps {
  messages: ParsedMessage[];
}

export function HistoryTree({ messages }: HistoryTreeProps) {
  const { t } = useLanguage();
  const [isExpanded, setIsExpanded] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded} className="mb-2">
      <CollapsibleTrigger className="flex w-full items-center gap-1.5 rounded py-1 text-left transition-colors hover:bg-muted">
        {isExpanded ? (
          <ChevronDown className="size-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-3 shrink-0 text-muted-foreground" />
        )}
        <span className="text-[10px] uppercase tracking-[0.5px] text-muted-foreground">
          {messages.length === 1
            ? t("HISTORY ({count} message)", { count: messages.length })
            : t("HISTORY ({count} messages)", { count: messages.length })}
        </span>
      </CollapsibleTrigger>

      <CollapsibleContent keepMounted className="mt-1 border-l border-border pl-4">
        {messages.map((msg, index) => (
          <SimpleMessageBlock
            key={index}
            label={msg.role.toUpperCase()}
            content={msg.content}
            toolCalls={msg.toolCalls}
            isCompact={true}
          />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
