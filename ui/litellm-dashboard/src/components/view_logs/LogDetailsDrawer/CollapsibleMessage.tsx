import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface CollapsibleMessageProps {
  label: string;
  content?: string;
  defaultExpanded?: boolean;
}

export function CollapsibleMessage({
  label,
  content,
  defaultExpanded = false,
}: CollapsibleMessageProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const charCount = content?.length || 0;

  if (!content || charCount === 0) {
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
          {label}
        </span>
        <span className="text-[10px] text-muted-foreground">
          ({charCount.toLocaleString()} chars)
        </span>
      </div>

      <div
        className="overflow-hidden transition-[max-height,opacity] duration-200 ease-out"
        style={{
          maxHeight: isExpanded ? "2000px" : "0px",
          opacity: isExpanded ? 1 : 0,
        }}
      >
        <div className="pl-4 text-[13px] leading-7 text-foreground border-l border-border whitespace-pre-wrap break-words">
          {content}
        </div>
      </div>
    </div>
  );
}
