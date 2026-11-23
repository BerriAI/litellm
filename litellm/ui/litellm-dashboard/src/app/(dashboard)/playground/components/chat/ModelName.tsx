"use client";

import { Copy, Split } from "lucide-react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { toast } from "@/lib/toast";

export function copyText(value: string, label: string, t: TFunction = i18n.t.bind(i18n)): void {
  void navigator.clipboard.writeText(value).then(
    () => toast.success(t("playground.chat.modelName.copied", { label })),
    () => toast.error(t("playground.chat.modelName.copyFailed")),
  );
}

export function CopyIconButton({ value, label }: { value: string; label: string }) {
  const { t } = useTranslation();
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={t("playground.chat.modelName.copy", { label })}
            onClick={() => copyText(value, label, t)}
          />
        }
      >
        <Copy />
      </TooltipTrigger>
      <TooltipContent>{t("playground.chat.modelName.copy", { label })}</TooltipContent>
    </Tooltip>
  );
}

export function LongValue({
  value,
  detail,
  copyLabel,
  className,
  align = "start",
}: {
  value: string;
  detail: string;
  copyLabel: string;
  className?: string;
  align?: "start" | "end";
}) {
  return (
    <HoverCard>
      <HoverCardTrigger
        delay={300}
        render={<span tabIndex={0} className={cn("min-w-0 cursor-default truncate font-mono", className)} />}
      >
        {value}
      </HoverCardTrigger>
      <HoverCardContent align={align} className="w-80">
        <div className="flex items-start gap-2">
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <span className="text-xs text-muted-foreground">{detail}</span>
            <span className="font-mono text-sm wrap-anywhere">{value}</span>
          </div>
          <CopyIconButton value={value} label={copyLabel} />
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

export function FallbackBadge() {
  const { t } = useTranslation();
  return (
    <Badge variant="outline" className="border-warning/20 bg-warning/10 text-warning">
      <Split data-icon="inline-start" />
      {t("playground.chat.modelName.fallback")}
    </Badge>
  );
}
