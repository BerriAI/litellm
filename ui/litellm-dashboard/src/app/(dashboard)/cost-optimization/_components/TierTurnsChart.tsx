"use client";

import React, { useState } from "react";
import { routingTiers, routingSpend, trafficShare, type RoutingUsage } from "@/components/UsagePage/routingUsage";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { hydrateTierLabels } from "@/components/add_model/build_complexity_router_config";
import {
  TIER_KEYS,
  effectiveTierLabel,
  type ComplexityTierLabels,
  type ComplexityTiers,
} from "@/components/add_model/ComplexityRouterConfig";
import { normalizeTierModels } from "@/components/add_model/complexity_router_tiers";
import { chartColorValue, DEFAULT_COLOR_CYCLE, DonutChart } from "@/components/shared/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError } from "@/lib/http/client";

import { viewGroup, type BenchmarkView } from "./autoRouterBenchmarks";

const safeParse = (value: string): unknown => {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const asRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? safeParse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

const isComplexityTier = (tier: string): tier is keyof ComplexityTiers =>
  (TIER_KEYS as readonly string[]).includes(tier);

export const tierDisplayLabel = (tier: string, tierLabels: ComplexityTierLabels | undefined): string =>
  isComplexityTier(tier) ? effectiveTierLabel(tier, tierLabels) : tier;

const CONFIG_KEY_BY_ROUTER_TYPE: Record<string, keyof NonNullable<AutoRouterDeployment["litellm_params"]>> = {
  complexity: "complexity_router_config",
  quality: "quality_router_config",
  auto_router: "auto_router_config",
  adaptive: "adaptive_router_config",
};

const deploymentFor = (
  routerName: string,
  routerType: string,
  autoRouters: readonly AutoRouterDeployment[],
): AutoRouterDeployment | undefined => {
  const configKey = CONFIG_KEY_BY_ROUTER_TYPE[routerType];
  if (!configKey) return undefined;
  return autoRouters.find((d) => d.model_name === routerName && d.litellm_params?.[configKey]);
};

const tierLabelsFor = (
  routerName: string,
  routerType: string,
  autoRouters: readonly AutoRouterDeployment[],
): ComplexityTierLabels | undefined => {
  const deployment = deploymentFor(routerName, routerType, autoRouters);
  if (!deployment) return undefined;
  const config = asRecord(deployment.litellm_params?.complexity_router_config);
  return hydrateTierLabels(config.tier_labels);
};

const configuredTiersFor = (
  routerName: string,
  routerType: string,
  autoRouters: readonly AutoRouterDeployment[],
): Record<string, unknown> => {
  const deployment = deploymentFor(routerName, routerType, autoRouters);
  if (!deployment) return {};
  const config = asRecord(deployment.litellm_params?.complexity_router_config);
  return asRecord(config.tiers);
};

interface TierTurnsChartProps {
  usage?: readonly RoutingUsage[];
  usageError?: unknown;
  view: BenchmarkView;
  autoRouters: readonly AutoRouterDeployment[];
}

const TierTurnsChart: React.FC<TierTurnsChartProps> = ({ view, autoRouters, usage, usageError }) => {
  const [mode, setMode] = useState<"turns" | "spend">("turns");
  const group = viewGroup(view);
  const entries = Object.entries(group?.tier_turns ?? {}).filter(([, turns]) => turns > 0);
  if (!group || (entries.length === 0 && !usage?.length)) return null;

  const tierLabels = tierLabelsFor(group.router_name, group.router_type, autoRouters);
  const measured = usage !== undefined && usage.length > 0;
  const spendUnavailable = !measured && (Boolean(usageError) || usage !== undefined);
  const configuredTiers = configuredTiersFor(group.router_name, group.router_type, autoRouters);
  const recordedTiers = routingTiers(usage ?? []);
  const tiers = [...new Set([...recordedTiers.map((row) => row.tier), ...Object.keys(configuredTiers)])];
  const slices = measured
    ? tiers.map((tier) => {
        const recorded = recordedTiers.find((row) => row.tier === tier);
        return {
          tier: tierDisplayLabel(tier ?? "Default / no tier", tierLabels),
          turns: recorded?.requests ?? 0,
          spend: recorded?.spend ?? 0,
          models: tier === null ? [] : normalizeTierModels(configuredTiers[tier]),
          modelUsage: recorded?.models ?? [],
        };
      })
    : entries.map(([tier, turns]) => ({
        tier: tierDisplayLabel(tier, tierLabels),
        turns,
        spend: 0,
        models: normalizeTierModels(configuredTiers[tier]),
        modelUsage: [],
      }));
  const hasSpend = measured && slices.some((slice) => slice.spend > 0);
  const category = hasSpend ? mode : "turns";
  const total = slices.reduce((sum, slice) => sum + slice[category], 0);
  const requests = slices.reduce((sum, slice) => sum + slice.turns, 0);
  const colors = slices.map((_, idx) => DEFAULT_COLOR_CYCLE[idx % DEFAULT_COLOR_CYCLE.length]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <CardTitle>Routing by tier</CardTitle>
          {measured && (
            <div className="flex rounded-lg bg-muted p-1" role="group" aria-label="Routing distribution">
              {(["turns", "spend"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={category === value}
                  disabled={value === "spend" && !hasSpend}
                  onClick={() => setMode(value)}
                  className={`rounded-md px-3 py-1 text-sm transition-colors disabled:opacity-50 ${category === value ? "bg-card text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"}`}
                >
                  {value === "turns" ? "Traffic" : "Spend"}
                </button>
              ))}
            </div>
          )}
        </div>
        <p className="text-sm text-muted-foreground">
          {measured
            ? "Requests and model spend from retained logs in this period. Excludes classifier and shadow-evaluation calls."
            : "Turns each tier served. Turns the classifier sent to the default model belong to no tier and are not counted here, so this can total less than the router's turns."}
          {spendUnavailable &&
            (usageError instanceof ApiError && usageError.status === 400
              ? ` ${usageError.message}.`
              : " Model spend is unavailable for this range.")}
          {measured && !hasSpend && " No model spend recorded; showing traffic."}
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 items-center gap-6 lg:grid-cols-2">
          <DonutChart
            className="h-80"
            data={slices}
            index="tier"
            category={category}
            colors={colors}
            valueFormatter={category === "spend" ? routingSpend : (value) => value.toLocaleString()}
            showLabel
            label={
              category === "spend"
                ? routingSpend(total)
                : `${total.toLocaleString()} total ${measured ? "requests" : "turns"}`
            }
          />
          <ul className="flex flex-col gap-6">
            {slices.map((slice, idx) => (
              <li key={slice.tier} className="flex items-start gap-2">
                <span
                  className="mt-1.5 h-2 w-2 shrink-0 rounded-full ring-4 ring-white"
                  style={{ backgroundColor: chartColorValue(colors[idx]) }}
                />
                <div className="min-w-0">
                  <p className="text-sm text-muted-foreground">
                    {slice.tier} {trafficShare(slice[category], total)}
                    {measured && ` · ${routingSpend(slice.spend)}`}
                  </p>
                  {measured && slice.models.length > 0 && (
                    <p className="text-xs break-words text-muted-foreground">Configured: {slice.models.join(", ")}</p>
                  )}
                  {measured && slice.modelUsage.length === 0 && (
                    <p className="text-xs text-muted-foreground">No retained requests for this tier</p>
                  )}
                  {measured
                    ? slice.modelUsage.map((model) => (
                        <p key={model.model} className="text-xs break-words text-muted-foreground">
                          {model.model} · {model.requests.toLocaleString()}{" "}
                          {model.requests === 1 ? "request" : "requests"} ({trafficShare(model.requests, requests)}) ·{" "}
                          {routingSpend(model.spend)}
                        </p>
                      ))
                    : slice.models.length > 0 && (
                        <p className="text-xs break-words text-muted-foreground">{slice.models.join(", ")}</p>
                      )}
                </div>
              </li>
            ))}
          </ul>
        </div>
      </CardContent>
    </Card>
  );
};

export default TierTurnsChart;
