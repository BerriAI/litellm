"use client";

import { Copy, Split } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { toast } from "@/lib/toast";

export function copyText(value: string, label: string): void {
  void navigator.clipboard.writeText(value).then(
    () => toast.success(`${label} copied`),
    () => toast.error("Unable to copy to clipboard"),
  );
}

export function CopyIconButton({ value, label }: { value: string; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button variant="ghost" size="icon-xs" aria-label={`Copy ${label}`} onClick={() => copyText(value, label)} />
        }
      >
        <Copy />
      </TooltipTrigger>
      <TooltipContent>Copy {label}</TooltipContent>
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
  return (
    <Badge variant="outline" className="border-warning/20 bg-warning/10 text-warning">
      <Split data-icon="inline-start" />
      Fallback
    </Badge>
  );
}
