/**
 * SectionHeader - Datadog-style header with icon, label, metrics, and copy
 */

import { ChevronDown, ChevronUp, Copy, MessageSquare } from "lucide-react";
import { useTranslation } from "react-i18next";
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

const SUMMARY_CLASSES = "flex flex-1 items-center gap-4";

export function SectionHeader({
  type,
  tokens,
  cost,
  onCopy,
  isCollapsed,
  onToggleCollapse,
  turnCount,
}: SectionHeaderProps) {
  const { t } = useTranslation();
  const summary = (
    <>
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
        <span className="text-sm font-medium">
          {type === "input" ? t("logs.detail.sectionHeader.input") : t("logs.detail.sectionHeader.output")}
        </span>
      </div>

      {tokens !== undefined && (
        <span className="text-xs text-muted-foreground">
          {t("logs.detail.sectionHeader.tokens", { value: tokens.toLocaleString() })}
        </span>
      )}

      {cost !== undefined && (
        <span className="text-xs text-muted-foreground">
          {t("logs.detail.sectionHeader.cost", { value: cost.toFixed(6) })}
        </span>
      )}

      {turnCount !== undefined && turnCount > 0 && (
        <span className="text-xs text-muted-foreground">
          {t("logs.detail.sectionHeader.turns", { count: turnCount })}
        </span>
      )}
    </>
  );

  return (
    <div
      className={cn(
        "flex items-center justify-between bg-muted px-4 py-2.5 transition-colors",
        isCollapsed ? "border-b-0" : "border-b border-border",
      )}
    >
      {onToggleCollapse ? (
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-expanded={!isCollapsed}
          className={cn(SUMMARY_CLASSES, "-mx-2 cursor-pointer rounded-md px-2 py-1 text-left hover:bg-accent")}
        >
          {summary}
        </button>
      ) : (
        <div className={SUMMARY_CLASSES}>{summary}</div>
      )}

      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={
                type === "input" ? t("logs.detail.sectionHeader.copyInput") : t("logs.detail.sectionHeader.copyOutput")
              }
              onClick={(e) => {
                e.stopPropagation();
                onCopy();
              }}
            />
          }
        >
          <Copy />
        </TooltipTrigger>
        <TooltipContent>{t("logs.detail.sectionHeader.copy")}</TooltipContent>
      </Tooltip>
    </div>
  );
}
