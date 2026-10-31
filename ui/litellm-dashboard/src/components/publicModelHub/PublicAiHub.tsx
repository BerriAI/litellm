"use client";

import type { ReactNode } from "react";

import Navbar from "@/components/navbar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/contexts/ThemeContext";

import { HubHeader } from "./HubHeader";
import { HubNotEnabled } from "./HubEmpty";
import { ModelsPane } from "./ModelsPane";
import { AgentsPane, McpPane, SkillsPane } from "./ResourcePanes";
import { type HubData, useHubData } from "./useHubData";
import { type HubTab, useHubUrlState } from "./useHubUrlState";
import { useRequirePublicHubAuth } from "./useRequirePublicHubAuth";

function TabLabel({ children, count }: { children: ReactNode; count: number | undefined }) {
  return (
    <>
      {children}
      <span className="text-xs text-muted-foreground tabular-nums">{count ?? "-"}</span>
    </>
  );
}

const hubStatus = (hub: HubData) => {
  if (hub.models.isLoading) return "loading";
  return hub.models.error ? "unavailable" : "operational";
};

function HubTabs({ hub }: { hub: HubData }) {
  const { tab, setTab, open, setOpen } = useHubUrlState();
  const paneProps = { selected: open, onSelect: setOpen };
  return (
    <Tabs value={tab} onValueChange={(next) => setTab(next as HubTab)} className="gap-4">
      <TabsList className="w-full sm:w-fit">
        <TabsTrigger value="models" className="px-1.5 sm:px-2">
          <TabLabel count={hub.modelTotal}>Models</TabLabel>
        </TabsTrigger>
        <TabsTrigger value="agents" className="px-1.5 sm:px-2">
          <TabLabel count={hub.agents.data?.length}>Agents</TabLabel>
        </TabsTrigger>
        <TabsTrigger value="mcp" className="px-1.5 sm:px-2">
          <TabLabel count={hub.mcp.data?.length}>
            <span className="sm:hidden">MCP</span>
            <span className="hidden sm:inline">MCP servers</span>
          </TabLabel>
        </TabsTrigger>
        <TabsTrigger value="skills" className="px-1.5 sm:px-2">
          <TabLabel count={hub.skills.data?.length}>Skills</TabLabel>
        </TabsTrigger>
      </TabsList>
      <TabsContent value="models">
        <ModelsPane hub={hub} {...paneProps} />
      </TabsContent>
      <TabsContent value="agents">
        <AgentsPane rows={hub.agents.data} loading={hub.agents.isPending} {...paneProps} />
      </TabsContent>
      <TabsContent value="mcp">
        <McpPane rows={hub.mcp.data} loading={hub.mcp.isPending} {...paneProps} />
      </TabsContent>
      <TabsContent value="skills">
        <SkillsPane rows={hub.skills.data} loading={hub.skills.isPending} {...paneProps} />
      </TabsContent>
    </Tabs>
  );
}

function PublicAiHubContent() {
  useRequirePublicHubAuth();
  const hub = useHubData();
  const status = hubStatus(hub);
  return (
    <div className="flex w-full flex-col gap-6 p-4 sm:p-6">
      <HubHeader info={hub.info.data} status={status} />
      {status === "unavailable" ? <HubNotEnabled /> : <HubTabs hub={hub} />}
    </div>
  );
}

export default function PublicAiHub({ accessToken }: { accessToken: string | null }) {
  return (
    <ThemeProvider accessToken={accessToken}>
      <TooltipProvider>
        <div className="min-h-screen bg-background">
          <Navbar accessToken={accessToken} isPublicPage />
          <PublicAiHubContent />
        </div>
      </TooltipProvider>
    </ThemeProvider>
  );
}
