/**
 * SectionHeader - Datadog-style header with icon, label, metrics, and copy
 */

import { ChevronDown, ChevronUp, Copy, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";

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
        "flex items-center justify-between bg-muted px-4 py-2.5 transition-colors",
        isCollapsed ? "border-b-0" : "border-b border-border",
        onToggleCollapse ? "cursor-pointer hover:bg-accent" : "cursor-default",
      )}
    >
      <div className="flex items-center gap-4">
        {onToggleCollapse &&
          (isCollapsed ? (
            <ChevronDown className="size-2.5 text-muted-foreground" />
          ) : (
            <ChevronUp className="size-2.5 text-muted-foreground" />
          ))}

        <div className="flex items-center gap-2">
          {type === "input" ? (
            <MessageSquare className="size-3.5 text-muted-foreground" />
          ) : (
            <span className="text-sm opacity-60 grayscale">✨</span>
          )}
          <span className="text-sm font-medium">{type === "input" ? "Input" : "Output"}</span>
        </div>

        {tokens !== undefined && (
          <span className="text-xs text-muted-foreground">Tokens: {tokens.toLocaleString()}</span>
        )}

        {cost !== undefined && <span className="text-xs text-muted-foreground">Cost: ${cost.toFixed(6)}</span>}

        {turnCount !== undefined && turnCount > 0 && (
          <span className="text-xs text-muted-foreground">Turns: {turnCount}</span>
        )}
      </div>

      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Copy"
              onClick={(e) => {
                e.stopPropagation();
                onCopy();
              }}
            />
          }
        >
          <Copy />
        </TooltipTrigger>
        <TooltipContent>Copy</TooltipContent>
      </Tooltip>
    </div>
  );
}
