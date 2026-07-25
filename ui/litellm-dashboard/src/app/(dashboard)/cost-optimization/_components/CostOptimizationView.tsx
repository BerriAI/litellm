"use client";

import React from "react";
import { Info, PiggyBank } from "lucide-react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";
import UsageTab from "./UsageTab";
import PromptCompressionTab from "./PromptCompressionTab";
import AutorouterTab from "./AutorouterTab";
import PromptCachingTab from "./PromptCachingTab";
import { useDailyActivityRange } from "./useDailyActivityRange";

interface CostOptimizationViewProps {
  accessToken: string | null;
  userId: string | null;
  userRole: string;
}

const CostOptimizationView: React.FC<CostOptimizationViewProps> = ({ accessToken, userId, userRole }) => {
  const activity = useDailyActivityRange(accessToken, userId, userRole);
  const { onTabChange, hasVisited } = useVisitedTabs("usage");

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

      <div role="alert" className="flex gap-2.5 rounded-lg border bg-card px-4 py-3 text-sm">
        <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
        <div>
          <p className="font-medium text-foreground">This is an experimental dashboard</p>
          <p className="mt-0.5 text-muted-foreground">
            Have feedback? Join the discussion{" "}
            <a
              href="https://github.com/BerriAI/litellm/discussions/32172"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-4 hover:text-foreground"
            >
              here
            </a>
          </p>
        </div>
      </div>

      <Tabs defaultValue="usage" onValueChange={onTabChange}>
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="usage" className="flex-none rounded-none px-4 py-2">
            Usage
          </TabsTrigger>
          <TabsTrigger value="compression" className="flex-none rounded-none px-4 py-2">
            Prompt Compression
          </TabsTrigger>
          <TabsTrigger value="autorouter" className="flex-none rounded-none px-4 py-2">
            Autorouter
          </TabsTrigger>
          <TabsTrigger value="caching" className="flex-none rounded-none px-4 py-2">
            Prompt Caching
          </TabsTrigger>
        </TabsList>

        <TabsContent keepMounted={hasVisited("usage")} value="usage" className="pt-4">
          <UsageTab accessToken={accessToken} activity={activity} />
        </TabsContent>
        <TabsContent keepMounted={hasVisited("compression")} value="compression" className="pt-4">
          <PromptCompressionTab accessToken={accessToken} />
        </TabsContent>
        <TabsContent keepMounted={hasVisited("autorouter")} value="autorouter" className="pt-4">
          <AutorouterTab accessToken={accessToken} userId={userId} userRole={userRole} />
        </TabsContent>
        <TabsContent keepMounted={hasVisited("caching")} value="caching" className="pt-4">
          <PromptCachingTab accessToken={accessToken} activity={activity} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CostOptimizationView;
