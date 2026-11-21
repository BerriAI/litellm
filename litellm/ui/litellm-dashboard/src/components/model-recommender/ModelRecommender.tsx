"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ArrowUpRight, Brain, Gauge, Loader2, SlidersHorizontal, Sparkles, Wallet } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  buildRequest,
  FOCUS,
  PRIORITIES,
  recommenderParsers,
  requestRecommendations,
  type RecommendationResponse,
} from "./api";
import { RecommendationCard } from "./RecommendationCard";

const icons = { intelligence: Brain, speed: Gauge, cost: Wallet };
const REQUIREMENTS = ["reasoning", "vision", "function_calling"] as const;
const IMPORTANCE_LEVELS = [0, 1, 2, 3] as const;

export default function ModelRecommender() {
  const { t } = useTranslation(undefined, { keyPrefix: "modelRecommender" });
  const { accessToken } = useAuthorized();
  const [state, setState] = useQueryStates(recommenderParsers);
  const [result, setResult] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const body = buildRequest(state);
  const fingerprint = JSON.stringify(body);
  const stale = submitted !== null && submitted !== fingerprint;
  useEffect(() => () => controller.current?.abort(), []);

  const recommend = async () => {
    if (!accessToken) return;
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setLoading(true);
    setError(null);
    try {
      const response = await requestRecommendations(body, accessToken, current.signal);
      if (current.signal.aborted) return;
      setResult(response);
      setSubmitted(fingerprint);
    } catch (failure) {
      if (!current.signal.aborted)
        setError(failure instanceof Error ? failure.message : "Could not reach your gateway. Please try again.");
    } finally {
      if (!current.signal.aborted) setLoading(false);
    }
  };

  return (
    <main className="mx-auto w-full max-w-[1440px] space-y-8 p-6 lg:p-10">
      <header className="flex flex-col justify-between gap-5 border-b pb-8 sm:flex-row sm:items-end">
        <div className="max-w-2xl space-y-3">
          <span className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] text-primary">
            <Sparkles className="size-3.5" />
            {t("badge")}
          </span>
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            {t("heading")}
            <br />
            <span className="text-muted-foreground">{t("subheading")}</span>
          </h1>
          <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">{t("intro")}</p>
        </div>
        <span className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="size-2 rounded-full bg-emerald-500" />
          {t("gatewayOnly")}
        </span>
      </header>
      <div className="grid items-start gap-8 xl:grid-cols-[340px_minmax(0,1fr)]">
        <Card className="xl:sticky xl:top-6">
          <CardContent className="space-y-7 pt-1">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="size-4 text-primary" />
              <h2 className="font-semibold">{t("brief")}</h2>
            </div>
            <div className="space-y-5">
              {PRIORITIES.map((kind) => {
                const Icon = icons[kind];
                return (
                  <fieldset key={kind} className="space-y-2.5">
                    <legend className="mb-2 flex items-center gap-2 text-sm font-medium capitalize">
                      <Icon className="size-4 text-muted-foreground" />
                      {t(`priorities.${kind}`)}
                    </legend>
                    <div className="grid grid-cols-4 gap-1 rounded-lg bg-muted/70 p-1">
                      {IMPORTANCE_LEVELS.map((value) => (
                        <button
                          type="button"
                          key={value}
                          aria-pressed={state[kind] === value}
                          onClick={() => setState({ [kind]: value })}
                          className={`rounded-md px-1 py-2 text-[10px] font-medium leading-tight transition-colors focus-visible:outline-2 focus-visible:outline-primary ${state[kind] === value ? "bg-background text-primary shadow-sm ring-1 ring-border" : "text-muted-foreground hover:bg-background/60"}`}
                        >
                          {t(`importance.${value}`)}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                );
              })}
            </div>
            <fieldset className="space-y-3 border-t pt-5">
              <legend className="mb-3 text-sm font-medium">{t("focusTitle")}</legend>
              <div className="flex flex-wrap gap-2">
                {FOCUS.map((focus) => (
                  <Button
                    key={focus}
                    size="sm"
                    variant={state.focus.includes(focus) ? "default" : "outline"}
                    aria-pressed={state.focus.includes(focus)}
                    onClick={() =>
                      setState({
                        focus: state.focus.includes(focus)
                          ? state.focus.filter((item) => item !== focus)
                          : [...state.focus, focus],
                      })
                    }
                    className="rounded-full capitalize"
                  >
                    {t(`focus.${focus}`)}
                  </Button>
                ))}
              </div>
            </fieldset>
            <fieldset className="space-y-4 border-t pt-5">
              <legend className="mb-4 text-sm font-medium">{t("requirements")}</legend>
              {REQUIREMENTS.map((key) => (
                <label
                  key={key}
                  className="flex items-center justify-between gap-3 text-sm text-muted-foreground"
                  htmlFor={`require-${key}`}
                >
                  {t(`capabilities.${key}`)}
                  <Switch
                    id={`require-${key}`}
                    checked={state[key]}
                    onCheckedChange={(checked) => setState({ [key]: checked })}
                  />
                </label>
              ))}
              <div className="space-y-2">
                <label htmlFor="min-context" className="text-xs text-muted-foreground">
                  {t("context")}
                </label>
                <Input
                  id="min-context"
                  type="number"
                  min={0}
                  placeholder={t("noMinimum")}
                  value={state.context ?? ""}
                  onChange={(event) => {
                    const value = event.target.value;
                    setState({ context: value === "" ? null : recommenderParsers.context.parse(value) });
                  }}
                />
              </div>
            </fieldset>
            <Button className="w-full" size="lg" disabled={loading || !accessToken} onClick={recommend}>
              {loading ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
              {loading ? t("finding") : t("recommend")}
            </Button>
            <p className="text-center text-[11px] text-muted-foreground">{t("noInference")}</p>
          </CardContent>
        </Card>
        <section className="min-w-0 space-y-5" aria-label={t("shortlist")} aria-busy={loading}>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">{t("shortlist")}</h2>
            {result && (
              <span className="text-xs text-muted-foreground">
                {t("counts", {
                  matching: result.candidates_considered - result.candidates_filtered,
                  filtered: result.candidates_filtered,
                })}
              </span>
            )}
          </div>
          {stale && (
            <p role="status" className="rounded-lg bg-muted px-4 py-3 text-sm">
              {t("stale")}
            </p>
          )}
          {error && (
            <div
              role="alert"
              className="rounded-xl border border-destructive/30 bg-destructive/5 p-5 text-sm text-destructive"
            >
              {error}
            </div>
          )}
          {loading && (
            <div role="status" className="space-y-4">
              <p className="text-sm text-muted-foreground">{t("loading")}</p>
              {[0, 1].map((index) => (
                <div key={index} className="h-64 animate-pulse rounded-xl bg-muted" />
              ))}
            </div>
          )}
          {!loading && !!result?.recommendations.length && (
            <>
              <RecommendationCard model={result.recommendations[0]} position={0} />
              <div className="grid gap-5 lg:grid-cols-2">
                {result.recommendations.slice(1).map((model, index) => (
                  <RecommendationCard key={model.model_group} model={model} position={index + 1} />
                ))}
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">{t("relativeScores")}</p>
            </>
          )}
          {!loading && !result?.recommendations.length && (
            <div className="flex min-h-[420px] flex-col items-center justify-center rounded-2xl border border-dashed bg-muted/20 px-8 text-center">
              <div className="mb-6 flex size-16 items-center justify-center rounded-2xl border bg-background shadow-sm">
                <Sparkles className="size-7 text-primary" />
              </div>
              <h3 className="text-xl font-semibold tracking-tight">{result ? t("noMatchTitle") : t("emptyTitle")}</h3>
              <p className="mt-3 max-w-sm text-sm leading-relaxed text-muted-foreground">
                {result ? t("noMatchDescription") : t("emptyDescription")}
              </p>
            </div>
          )}
          <footer className="space-y-2 border-t pt-5 text-[11px] leading-relaxed text-muted-foreground">
            <p>
              <a
                href="https://epoch.ai/benchmarks"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 underline underline-offset-4"
              >
                {t("epoch")} <ArrowUpRight className="size-3" />
              </a>
              {" · "}
              <a
                href="https://creativecommons.org/licenses/by/4.0/"
                target="_blank"
                rel="noreferrer"
                className="underline underline-offset-4"
              >
                {t("license")}
              </a>
              {". "}
              {t("sources")}
            </p>
            {result?.data_sources.map((source) => (
              <p key={source.name}>
                <span className="font-medium">{source.name}</span> ·{" "}
                {source.available ? t("asOf", { date: source.as_of ?? t("thisRequest") }) : t("unavailable")}.{" "}
                {source.description}
              </p>
            ))}
          </footer>
        </section>
      </div>
    </main>
  );
}
