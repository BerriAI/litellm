"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Copy, Play, Plus } from "lucide-react";
import { Panel, Stat } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import { Logo } from "@/components/molecules/logo/Logo";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";
import { compactNumber, metricValue, PRIORITIES, type Recommendation } from "./api";

export function RecommendationCard({ model, position }: { model: Recommendation; position: number }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const provider = model.providers[0] ?? "";
  const onGateway = model.on_gateway !== false;
  const destination = onGateway
    ? `playground?model=${encodeURIComponent(model.model_group)}`
    : `models-and-endpoints?tab=add&provider=${encodeURIComponent(provider)}&litellm_model=${encodeURIComponent(model.model_group)}`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(model.model_group);
      setCopied(true);
      setCopyError(false);
    } catch {
      setCopyError(true);
    }
  };
  return (
    <Panel
      className="@container"
      title={t("rankPick", { rank: position + 1 })}
      action={!onGateway && <Badge variant="secondary">{t("notOnGateway")}</Badge>}
      bodyClassName="gap-5"
    >
      <div className="flex min-w-0 items-center gap-3">
        <Logo provider={provider} className="size-8 shrink-0" />
        <div className="min-w-0">
          <h2 className="break-words text-xl font-semibold tracking-tight">{model.model_group}</h2>
          <p className="text-xs text-muted-foreground">{t("scoreLabel", { score: Math.round(model.score) })}</p>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-5 border-y py-4 @lg:grid-cols-4">
        <Stat
          label={t("priorities.intelligence")}
          value={
            model.signals.intelligence.raw == null
              ? t("noData")
              : `${Math.round(model.signals.intelligence.raw * 100)}%`
          }
        />
        <Stat
          label={t("priorities.speed")}
          value={metricValue("speed", model.latency_seconds)}
          hint={
            model.output_tokens_per_second == null
              ? undefined
              : t("throughput", { value: Math.round(model.output_tokens_per_second) })
          }
        />
        <Stat
          label={t("costPerMillion")}
          value={model.signals.cost.raw == null ? t("noData") : `$${model.signals.cost.raw.toFixed(2)}`}
        />
        <Stat
          label={t("contextLabel")}
          value={model.context_window == null ? t("noData") : compactNumber(model.context_window)}
        />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {PRIORITIES.map((kind) => (
          <div key={kind} className="space-y-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{t(`priorities.${kind}`)}</span>
              <span>
                {model.signals[kind].unknown ? t("noData") : Math.round(model.signals[kind].normalized * 100)}
              </span>
            </div>
            <div
              className="h-1 overflow-hidden rounded-full bg-muted"
              role="meter"
              aria-label={t("relativeRating", { kind })}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(model.signals[kind].normalized * 100)}
            >
              <div className="h-full bg-[#2b3fd6]" style={{ width: `${model.signals[kind].normalized * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="space-y-2 text-sm">
        <h3 className="font-medium">{t("why")}</h3>
        {model.reasons.slice(0, 3).map((reason) => (
          <p key={reason} className="text-muted-foreground">
            {reason}
          </p>
        ))}
      </div>
      <div className="mt-auto flex flex-wrap gap-2">
        <Button nativeButton={false} role="link" render={<a href={uiHref(destination)} />}>
          {onGateway ? <Play /> : <Plus />}
          {onGateway ? t("playground") : t("addModel")}
        </Button>
        <Button variant="outline" onClick={copy}>
          {copied ? <Check /> : <Copy />}
          {copied ? t("copied") : t("copy")}
        </Button>
      </div>
      {copyError && (
        <p role="alert" className="text-xs text-destructive">
          {t("copyFailed")}
        </p>
      )}
    </Panel>
  );
}
