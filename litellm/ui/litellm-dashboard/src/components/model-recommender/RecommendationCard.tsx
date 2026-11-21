"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import Image from "next/image";
import { Check, Copy, Play, Trophy } from "lucide-react";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { metricValue, PRIORITIES, type Recommendation } from "./api";

export function RecommendationCard({ model, position }: { model: Recommendation; position: number }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const router = useRouter();
  const hero = position === 0;
  const provider = getProviderLogoAndName(model.providers[0] ?? "");
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
    <Card
      className={`relative overflow-hidden ${hero ? "border border-primary/25 bg-gradient-to-br from-primary/5 via-card to-card shadow-lg shadow-primary/5" : ""}`}
    >
      <CardContent className={`space-y-6 ${hero ? "pt-2 sm:p-8" : "pt-1"}`}>
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-muted-foreground">
            {hero && <Trophy className="size-4 text-primary" />}
            {hero ? t("strongest") : t("alternative", { position: position + 1 })}
          </span>
          <div
            className="flex size-16 shrink-0 flex-col items-center justify-center rounded-full border-4 border-primary/25 bg-background"
            aria-label={t("scoreLabel", { score: model.score })}
          >
            <span className="text-lg font-semibold tabular-nums leading-none">{Math.round(model.score)}</span>
            <span className="mt-1 text-[9px] uppercase tracking-wide text-muted-foreground">{t("match")}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {provider.logo && (
            <Image
              src={provider.logo}
              width={40}
              height={40}
              alt={provider.displayName}
              className="size-10 rounded-lg bg-white p-1.5 object-contain ring-1 ring-border"
            />
          )}
          <div className="min-w-0">
            <h3 className={`break-words font-semibold tracking-tight ${hero ? "text-3xl" : "text-xl"}`}>
              {model.model_group}
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">{model.providers.join(" · ") || t("configured")}</p>
          </div>
        </div>
        <div className={`grid gap-5 ${hero ? "sm:grid-cols-3" : ""}`}>
          {PRIORITIES.map((kind) => {
            const signal = model.signals[kind];
            return (
              <div key={kind} className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="font-medium capitalize">{t(`priorities.${kind}`)}</span>
                  <span className="text-muted-foreground">
                    {signal.unknown ? t("noData") : `${Math.round(signal.normalized * 100)}/100`}
                  </span>
                </div>
                <div
                  role="meter"
                  aria-label={t("relativeRating", { kind: t(`priorities.${kind}`) })}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(signal.normalized * 100)}
                  className="h-1.5 overflow-hidden rounded-full bg-muted"
                >
                  <div
                    className={`h-full rounded-full ${signal.unknown ? "bg-muted-foreground/30" : "bg-primary"}`}
                    style={{ width: `${signal.normalized * 100}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground">{metricValue(kind, signal.raw)}</p>
              </div>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-2">
          {model.capabilities.reasoning && <Badge variant="secondary">{t("capabilities.reasoning")}</Badge>}
          {model.capabilities.vision && <Badge variant="secondary">{t("vision")}</Badge>}
          {model.capabilities.function_calling && (
            <Badge variant="secondary">{t("capabilities.function_calling")}</Badge>
          )}
          {model.context_window != null && (
            <Badge variant="outline">{t("contextTokens", { count: model.context_window })}</Badge>
          )}
          {model.output_tokens_per_second != null && (
            <Badge variant="outline">{t("throughput", { value: model.output_tokens_per_second.toFixed(1) })}</Badge>
          )}
        </div>
        <ul className="space-y-2 border-t border-border/60 pt-5">
          {model.reasons.map((reason) => (
            <li key={reason} className="flex gap-2 text-sm text-muted-foreground">
              <Check className="mt-0.5 size-4 shrink-0 text-primary" />
              {reason}
            </li>
          ))}
        </ul>
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={() => {
              window.sessionStorage.setItem("selectedModel", model.model_group);
              router.push("/playground");
            }}
          >
            <Play className="size-4" />
            {t("playground")}
          </Button>
          <Button variant="outline" onClick={copy}>
            <Copy className="size-4" />
            {copied ? t("copied") : t("copy")}
          </Button>
        </div>
        {copyError && (
          <p role="alert" className="text-sm text-destructive">
            {t("copyFailed")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
