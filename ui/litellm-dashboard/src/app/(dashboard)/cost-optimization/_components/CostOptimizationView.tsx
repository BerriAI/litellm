"use client";

import React from "react";
import { Info, PiggyBank } from "lucide-react";

import useCan from "@/app/(dashboard)/hooks/useCan";
import PaginationStatusAlerts from "@/components/shared/PaginationStatusAlerts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/shared/PageHeader";
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
    <main className="w-full p-8">
      <Tabs defaultValue="usage" onValueChange={handleTabChange} className="gap-6">
        <PageHeader
          icon={<PiggyBank />}
          title="Cost Optimization"
          subtitle="Track and configure the mechanisms that save you money: prompt compression and prompt caching. Auto routers live under Models + Endpoints, on the Auto-Routers tab"
          tabs={({ leadingControls }) => (
            <TabsList
              variant="line"
              className="gap-0 p-0 [&>[data-slot=tabs-trigger]+[data-slot=tabs-trigger]]:ml-[22px]"
            >
              {leadingControls}
              <TabsTrigger value="usage" className="flex-none px-0 py-[7px] data-active:font-semibold">
                Overall
              </TabsTrigger>
              {canViewProxyWideCostData && (
                <>
                  <TabsTrigger value="compression" className="flex-none px-0 py-[7px] data-active:font-semibold">
                    Prompt Compression
                  </TabsTrigger>
                  <TabsTrigger value="caching" className="flex-none px-0 py-[7px] data-active:font-semibold">
                    Prompt Caching
                  </TabsTrigger>
                  <TabsTrigger value="autorouter-usage" className="flex-none px-0 py-[7px] data-active:font-semibold">
                    Auto-Router
                  </TabsTrigger>
                </>
              )}
            </TabsList>
          )}
        />

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

        <PaginationStatusAlerts
          isFetchingMore={activity.isFetchingMore}
          cancelled={activity.cancelled}
          progress={activity.progress}
          cancel={activity.cancel}
        />
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
              <AutoRouterBenchmarksTab accessToken={accessToken} activity={activity} />
            </TabsContent>
          </>
        )}
      </Tabs>
    </main>
  );
};

export default CostOptimizationView;
