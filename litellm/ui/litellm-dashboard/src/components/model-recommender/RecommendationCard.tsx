"use client";

import { useId, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import Image from "next/image";
import { Check, Copy, Play, Trophy } from "lucide-react";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { compactNumber, metricValue, PRIORITIES, type Recommendation } from "./api";

export function RecommendationCard({ model, position }: { model: Recommendation; position: number }) {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const router = useRouter();
  const ringId = useId();
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
      className={`relative overflow-hidden rounded-2xl py-0 ${hero ? "border-blue-300/70 bg-linear-to-br from-blue-50 via-card to-cyan-50/70 shadow-xl shadow-blue-500/10 ring-1 ring-blue-400/10 dark:border-blue-500/40 dark:from-blue-950/60 dark:via-card dark:to-cyan-950/30" : "bg-linear-to-br from-card to-blue-50/40 shadow-sm dark:to-blue-950/20"}`}
    >
      {hero && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -right-16 -top-24 size-72 rounded-full bg-blue-400/15 blur-3xl dark:bg-blue-500/15"
        />
      )}
      <CardContent className={`relative ${hero ? "space-y-6 p-6 sm:p-7" : "space-y-5 p-5"}`}>
        <div className="flex items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            <Badge
              className={
                hero
                  ? "rounded-lg bg-blue-600 px-2.5 py-1 text-sm text-white shadow-md shadow-blue-500/20"
                  : "rounded-lg bg-blue-100 px-2.5 py-1 text-sm text-blue-700 dark:bg-blue-950 dark:text-blue-300"
              }
            >
              {`#${position + 1}`}
            </Badge>
            <span className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-blue-700 dark:text-blue-300">
              {hero && <Trophy className="size-3.5" />}
              {hero ? t("strongest") : t("alternative", { position: position + 1 })}
            </span>
          </div>
          <div
            role="meter"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={model.score}
            className={`relative flex shrink-0 flex-col items-center justify-center ${hero ? "size-24" : "size-16"}`}
            aria-label={t("scoreLabel", { score: model.score })}
          >
            <svg viewBox="0 0 100 100" className="absolute inset-0 size-full -rotate-90" aria-hidden="true">
              <defs>
                <linearGradient id={ringId} x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#2563eb" />
                  <stop offset="100%" stopColor="#22d3ee" />
                </linearGradient>
              </defs>
              <circle
                cx="50"
                cy="50"
                r="43"
                fill="none"
                strokeWidth="6"
                className="stroke-blue-200/60 dark:stroke-blue-900/60"
              />
              <circle
                cx="50"
                cy="50"
                r="43"
                fill="none"
                strokeWidth="6"
                stroke={`url(#${ringId})`}
                strokeLinecap="round"
                pathLength="100"
                strokeDasharray={`${model.score} 100`}
                className="transition-all duration-700 motion-reduce:transition-none"
              />
            </svg>
            <span className={`font-semibold tabular-nums leading-none ${hero ? "text-2xl" : "text-lg"}`}>
              {Math.round(model.score)}
            </span>
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
                <div className="flex justify-between gap-2 text-xs">
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
                  className="h-2.5 overflow-hidden rounded-full bg-blue-950/5 ring-1 ring-inset ring-blue-900/5 dark:bg-white/5 dark:ring-white/5"
                >
                  <div
                    className={`h-full rounded-full transition-all duration-700 motion-reduce:transition-none ${signal.unknown ? "bg-muted-foreground/30" : "bg-linear-to-r from-blue-600 via-blue-500 to-cyan-400"}`}
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
            <Badge variant="outline">{t("contextTokens", { value: compactNumber(model.context_window) })}</Badge>
          )}
          {model.output_tokens_per_second != null && (
            <Badge variant="outline">{t("throughput", { value: Math.round(model.output_tokens_per_second) })}</Badge>
          )}
        </div>
        <div className="space-y-3 border-t border-blue-200/60 pt-5 dark:border-blue-900/50">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-blue-700 dark:text-blue-300">
            {t("why")}
          </h4>
          <ul className="space-y-2">
            {model.reasons.map((reason) => (
              <li key={reason} className="flex gap-2 text-sm text-muted-foreground">
                <Check className="mt-0.5 size-4 shrink-0 text-blue-600 dark:text-blue-400" />
                {reason}
              </li>
            ))}
          </ul>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            className="bg-blue-600 text-white shadow-sm shadow-blue-500/20 hover:bg-blue-700"
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
