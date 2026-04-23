import { ToolCall } from "./prettyMessagesTypes";
import { SimpleToolCallBlock } from "./SimpleToolCallBlock";
import { cn } from "@/lib/utils";

interface SimpleMessageBlockProps {
  label: string;
  content?: string;
  toolCalls?: ToolCall[];
  isCompact?: boolean;
}

export function SimpleMessageBlock({
  label,
  content,
  toolCalls,
  isCompact = false,
}: SimpleMessageBlockProps) {
  const displayContent =
    content && content !== "null" && content.length > 0 ? content : null;
  const hasToolCalls = toolCalls && toolCalls.length > 0;

  if (!displayContent && !hasToolCalls) {
    return null;
  }

  return (
    <div className={cn(isCompact ? "mb-2" : "")}>
      <span className="block text-[10px] tracking-wider uppercase text-muted-foreground mb-0.5">
        {label}
      </span>

      {displayContent && (
        <div
          className={cn(
            "text-[13px] leading-7 text-foreground whitespace-pre-wrap break-words",
            hasToolCalls ? "mb-1.5" : "",
          )}
        >
          {displayContent}
        </div>
      )}

      {hasToolCalls && (
        <div>
          {toolCalls.map((tc, index) => (
            <SimpleToolCallBlock
              key={tc.id || index}
              tool={tc}
              compact={isCompact}
            />
          ))}
        </div>
      )}
    </div>
  );
}
