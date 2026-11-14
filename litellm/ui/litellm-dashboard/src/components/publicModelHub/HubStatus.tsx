"use client";

import { CircleCheck, CircleDashed, CircleX } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { Badge } from "@/components/ui/badge";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { cn } from "@/lib/cva.config";

import { healthOf, type HealthTone, relativeTime } from "./hubFormat";

const TONE_ICON = { success: CircleCheck, error: CircleX, neutral: CircleDashed } as const;
const TONE_CLASS = {
  success: "border-success/25 bg-success/10 text-success",
  error: "border-destructive/25 bg-destructive/10 text-destructive",
  neutral: "text-muted-foreground",
} as const;

export function StatusBadge({ tone, label }: { tone: HealthTone; label: string }) {
  const Icon = TONE_ICON[tone];
  return (
    <Badge variant="outline" className={cn("font-medium", TONE_CLASS[tone])}>
      <Icon data-icon="inline-start" />
      {label}
    </Badge>
  );
}

export function HealthBadge({ model }: { model: ModelGroupInfo }) {
  const { t } = useTranslation();
  const badge = <StatusBadge {...healthOf(model)} />;
  if (!model.health_status) return badge;
  return (
    <HoverCard>
      <HoverCardTrigger delay={150} render={<span tabIndex={0} className="inline-flex" />}>
        {badge}
      </HoverCardTrigger>
      <HoverCardContent align="end" className="w-56 p-3 text-xs">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1">
          <dt className="text-muted-foreground">{t("publicModelHub.healthBadge.responseTime")}</dt>
          <dd className="tabular-nums">
            {model.health_response_time
              ? t("publicModelHub.healthBadge.milliseconds", { value: Math.round(model.health_response_time) })
              : "-"}
          </dd>
          <dt className="text-muted-foreground">{t("publicModelHub.healthBadge.lastChecked")}</dt>
          <dd>{relativeTime(model.health_checked_at)}</dd>
        </dl>
      </HoverCardContent>
    </HoverCard>
  );
}
