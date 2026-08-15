"use client";

import React from "react";
import { Info, PiggyBank } from "lucide-react";

import useCan from "@/app/(dashboard)/hooks/useCan";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import UsageTab from "./UsageTab";
import PromptCompressionTab from "./PromptCompressionTab";
import PromptCachingTab from "./PromptCachingTab";
import AutoRouterBenchmarksTab from "./AutoRouterBenchmarksTab";
import { useDailyActivityRange } from "./useDailyActivityRange";

interface CostOptimizationViewProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

const CostOptimizationView: React.FC<CostOptimizationViewProps> = ({ accessToken, userId, userRole }) => {
  const activity = useDailyActivityRange(accessToken, userId, userRole);
  const canViewProxyWideCostData = useCan("viewProxyWideCostData");
  const [visitedTabs, setVisitedTabs] = React.useState<readonly string[]>(["usage"]);

  const handleTabChange = (value: unknown) => {
    if (typeof value !== "string") {
      return;
    }

    setVisitedTabs((currentTabs) => (currentTabs.includes(value) ? currentTabs : [...currentTabs, value]));
  };

  return (
    <div className="w-full space-y-6 p-6">
      <div>
        <div className="flex items-center gap-2">
          <PiggyBank className="size-6 text-primary" strokeWidth={1.75} />
          <h1 className="text-xl font-semibold text-foreground">Cost Optimization</h1>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Track and configure the mechanisms that save you money: prompt compression and prompt caching. Auto routers
          live under Models + Endpoints, on the Auto-Routers tab
        </p>
      </div>

      <div
        role="alert"
        className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-lg border border-border bg-muted/50 px-4 py-4"
      >
        <Info className="mt-0.5 size-5 text-primary" aria-hidden="true" />
        <p className="font-medium text-foreground">This is an experimental dashboard</p>
        <p className="col-start-2 text-sm text-muted-foreground">
          Have feedback? Join the discussion{" "}
          <a
            href="https://github.com/BerriAI/litellm/discussions/32168"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline underline-offset-2"
          >
            here
          </a>
        </p>
      </div>

      <Tabs defaultValue="usage" onValueChange={handleTabChange}>
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="usage" className="flex-none rounded-none px-4 py-2">
            Overall
          </TabsTrigger>
          {canViewProxyWideCostData && (
            <>
              <TabsTrigger value="compression" className="flex-none rounded-none px-4 py-2">
                Prompt Compression
              </TabsTrigger>
              <TabsTrigger value="caching" className="flex-none rounded-none px-4 py-2">
                Prompt Caching
              </TabsTrigger>
              <TabsTrigger value="autorouter-usage" className="flex-none rounded-none px-4 py-2">
                Auto-Router
              </TabsTrigger>
            </>
          )}
        </TabsList>

        <TabsContent value="usage" keepMounted={visitedTabs.includes("usage")}>
          <UsageTab accessToken={accessToken} activity={activity} />
        </TabsContent>
        {canViewProxyWideCostData && (
          <>
            <TabsContent value="compression" keepMounted={visitedTabs.includes("compression")}>
              <PromptCompressionTab accessToken={accessToken} />
            </TabsContent>
            <TabsContent value="caching" keepMounted={visitedTabs.includes("caching")}>
              <PromptCachingTab accessToken={accessToken} activity={activity} />
            </TabsContent>
            <TabsContent value="autorouter-usage" keepMounted={visitedTabs.includes("autorouter-usage")}>
              <AutoRouterBenchmarksTab accessToken={accessToken} />
            </TabsContent>
          </>
        )}
      </Tabs>
    </div>
  );
};

export default CostOptimizationView;
