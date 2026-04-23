import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChevronDown,
  ChevronUp,
  Copy,
  MessageSquare,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SectionHeaderProps {
  type: "input" | "output";
  tokens?: number;
  cost?: number;
  onCopy: () => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  turnCount?: number;
}

export function SectionHeader({
  type,
  tokens,
  cost,
  onCopy,
  isCollapsed,
  onToggleCollapse,
  turnCount,
}: SectionHeaderProps) {
  return (
    <div
      onClick={onToggleCollapse}
      className={cn(
        "flex items-center justify-between px-4 py-2.5 bg-muted/50 transition-colors",
        !isCollapsed && "border-b border-border",
        onToggleCollapse ? "cursor-pointer hover:bg-muted" : "",
      )}
    >
      <div className="flex items-center gap-4">
        {onToggleCollapse && (
          <div className="flex items-center">
            {isCollapsed ? (
              <ChevronDown className="h-2.5 w-2.5 text-muted-foreground" />
            ) : (
              <ChevronUp className="h-2.5 w-2.5 text-muted-foreground" />
            )}
          </div>
        )}

        <div className="flex items-center gap-2">
          {type === "input" ? (
            <MessageSquare className="h-3.5 w-3.5 text-muted-foreground" />
          ) : (
            <span className="text-sm grayscale opacity-60">✨</span>
          )}
          <span className="font-medium text-sm">
            {type === "input" ? "Input" : "Output"}
          </span>
        </div>

        {tokens !== undefined && (
          <span className="text-xs text-muted-foreground">
            Tokens: {tokens.toLocaleString()}
          </span>
        )}

        {cost !== undefined && (
          <span className="text-xs text-muted-foreground">
            Cost: ${cost.toFixed(6)}
          </span>
        )}

        {turnCount !== undefined && turnCount > 0 && (
          <span className="text-xs text-muted-foreground">
            Turns: {turnCount}
          </span>
        )}
      </div>

      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={(e) => {
                e.stopPropagation();
                onCopy();
              }}
            >
              <Copy className="h-3 w-3" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Copy</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
}
