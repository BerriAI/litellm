import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { ParsedMessage } from "./prettyMessagesTypes";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

interface HistoryTreeProps {
  messages: ParsedMessage[];
}

export function HistoryTree({ messages }: HistoryTreeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  return (
    <div className="mb-2">
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 cursor-pointer py-1 rounded hover:bg-muted transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        )}
        <span className="text-[10px] tracking-wider uppercase text-muted-foreground">
          HISTORY ({messages.length} message{messages.length !== 1 ? "s" : ""})
        </span>
      </div>

      <div
        className="overflow-hidden transition-[max-height,opacity] duration-200 ease-out"
        style={{
          maxHeight: isExpanded ? "2000px" : "0px",
          opacity: isExpanded ? 1 : 0,
        }}
      >
        <div className="pl-4 border-l border-border">
          {messages.map((msg, index) => (
            <SimpleMessageBlock
              key={index}
              label={msg.role.toUpperCase()}
              content={msg.content}
              toolCalls={msg.toolCalls}
              isCompact={true}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
