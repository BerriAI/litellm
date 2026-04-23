import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";
import { cn } from "@/lib/utils";
import { ParsedTool } from "./types";
import { ToolExpandedContent } from "./ToolExpandedContent";

interface ToolItemProps {
  tool: ParsedTool;
}

export function ToolItem({ tool }: ToolItemProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <div
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex items-center justify-between px-4 py-3 cursor-pointer transition-colors",
          expanded ? "bg-muted" : "bg-background",
        )}
      >
        <div className="flex items-center gap-2.5">
          <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-sm">
            {tool.index}. {tool.name}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <Badge
            className={cn(
              "text-xs",
              tool.called
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            {tool.called ? "called" : "not called"}
          </Badge>
          {expanded ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
        </div>
      </div>

      {expanded && (
        <div className="p-4 border-t border-border bg-background">
          <ToolExpandedContent tool={tool} />
        </div>
      )}
    </div>
  );
}
