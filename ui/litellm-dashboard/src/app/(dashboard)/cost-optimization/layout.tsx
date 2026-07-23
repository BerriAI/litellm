"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { PiggyBank, Info } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  costOptimizationTabHref,
  slugFromPathname,
  type CostOptimizationTabSlug,
} from "@/app/(dashboard)/cost-optimization/tabRoutes";

const BASE_TAB_KEY = "usage";

const ORDERED_KEYS: Array<"" | CostOptimizationTabSlug> = ["", "compression", "autorouter", "caching"];

const TAB_LABELS: Record<"" | CostOptimizationTabSlug, string> = {
  "": "Usage",
  compression: "Prompt Compression",
  autorouter: "Autorouter",
  caching: "Prompt Caching",
};

export default function CostOptimizationLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = ORDERED_KEYS.some((slug) => slug === activeSlug);
  const activeKey = isKnownSlug ? activeSlug || BASE_TAB_KEY : BASE_TAB_KEY;

  useEffect(() => {
    if (activeSlug !== "" && !isKnownSlug) {
      window.location.replace(costOptimizationTabHref(""));
    }
  }, [activeSlug, isKnownSlug]);

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

      <Tabs
        value={activeKey}
        onValueChange={(key) => router.push(costOptimizationTabHref(key === BASE_TAB_KEY ? "" : key))}
      >
        <TabsList variant="line">
          {ORDERED_KEYS.map((slug) => (
            <TabsTrigger key={slug || BASE_TAB_KEY} value={slug || BASE_TAB_KEY}>
              {TAB_LABELS[slug]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div>{children}</div>
    </div>
  );
}
