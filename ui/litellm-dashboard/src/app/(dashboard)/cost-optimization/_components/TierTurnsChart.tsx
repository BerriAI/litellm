"use client";

import React from "react";

import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { hydrateTierLabels } from "@/components/add_model/build_complexity_router_config";
import {
  TIER_KEYS,
  effectiveTierLabel,
  type ComplexityTierLabels,
  type ComplexityTiers,
} from "@/components/add_model/ComplexityRouterConfig";
import { normalizeTierModels } from "@/components/add_model/complexity_router_tiers";
import { chartColorValue, DonutChart, type ChartColor } from "@/components/shared/charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

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

const tierModelsFor = (
  tier: string,
  routerName: string,
  routerType: string,
  autoRouters: readonly AutoRouterDeployment[],
): string[] => {
  if (!isComplexityTier(tier)) return [];
  const deployment = deploymentFor(routerName, routerType, autoRouters);
  if (!deployment) return [];
  const config = asRecord(deployment.litellm_params?.complexity_router_config);
  const tiers = asRecord(config.tiers);
  return normalizeTierModels(tiers[tier]);
};

interface TierTurnsChartProps {
  view: BenchmarkView;
  autoRouters: readonly AutoRouterDeployment[];
}

const TIER_DONUT_COLORS: readonly ChartColor[] = ["#c7d2fe", "#1e293b", "#d4b483", "#87a878"];

const TierTurnsChart: React.FC<TierTurnsChartProps> = ({ view, autoRouters }) => {
  const group = viewGroup(view);
  const entries = Object.entries(group?.tier_turns ?? {}).filter(([, turns]) => turns > 0);
  if (!group || entries.length === 0) return null;

  const tierLabels = tierLabelsFor(group.router_name, group.router_type, autoRouters);
  const total = entries.reduce((sum, [, turns]) => sum + turns, 0);
  const slices = entries.map(([tier, turns]) => ({
    tier: tierDisplayLabel(tier, tierLabels),
    turns,
    models: tierModelsFor(tier, group.router_name, group.router_type, autoRouters),
  }));
  const colors = slices.map((_, idx) => TIER_DONUT_COLORS[idx % TIER_DONUT_COLORS.length]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Routing by tier</CardTitle>
        <p className="text-sm text-muted-foreground">
          Turns each tier served. Turns the classifier sent to the default model belong to no tier and are not counted
          here, so this can total less than the router&apos;s turns.
        </p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 items-center gap-6 lg:grid-cols-2">
          <DonutChart
            className="h-80"
            data={slices}
            index="tier"
            category="turns"
            colors={colors}
            valueFormatter={(value) => value.toLocaleString()}
            showLabel
            label={`${total.toLocaleString()} total turns`}
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
                    {slice.tier} {Math.round((100 * slice.turns) / total).toLocaleString()}%
                  </p>
                  {slice.models.length > 0 && (
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
