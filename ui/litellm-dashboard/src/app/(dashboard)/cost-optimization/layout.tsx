"use client";

import type { ReactNode } from "react";
import { PiggyBank, Info } from "lucide-react";
import { costOptimizationRoutes } from "@/app/(dashboard)/cost-optimization/tabRoutes";
import { useTabRouting } from "@/app/(dashboard)/hooks/useTabRouting";
import { TabRouteBar } from "@/app/(dashboard)/components/TabRouteBar";

const BASE_TAB_KEY = "usage";

const TABS = [
  { key: BASE_TAB_KEY, label: "Usage" },
  { key: "compression", label: "Prompt Compression" },
  { key: "autorouter", label: "Autorouter" },
  { key: "caching", label: "Prompt Caching" },
] as const;

export default function CostOptimizationLayout({ children }: { children: ReactNode }) {
  const { activeKey } = useTabRouting({
    routes: costOptimizationRoutes,
    baseTabKey: BASE_TAB_KEY,
    visibleKeys: costOptimizationRoutes.slugs,
  });

  return (
    <div className="w-full space-y-6 p-6">
      <div>
        <div className="flex items-center gap-2">
          <PiggyBank className="size-6 text-emerald-600" strokeWidth={1.75} />
          <h1 className="text-xl font-semibold text-foreground">Cost Optimization</h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Track and configure the mechanisms that save you money: prompt compression, prompt caching, and auto routing
        </p>
      </div>

      <div className="flex items-start gap-2 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-200">
        <Info className="mt-0.5 size-4 shrink-0" />
        <span>
          This is an experimental dashboard. Have feedback? Join the discussion{" "}
          <a
            href="https://github.com/BerriAI/litellm/discussions/32172"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            here
          </a>
        </span>
      </div>

      <TabRouteBar routes={costOptimizationRoutes} baseTabKey={BASE_TAB_KEY} activeKey={activeKey} tabs={TABS} />

      <div>{children}</div>
    </div>
  );
}
