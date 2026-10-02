"use client";

import { useEffect, useState } from "react";
import { Aperture } from "lucide-react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LensDemoContext, useLensDemo } from "@/components/lens/LensDemoContext";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { LensView } from "./LensView";
import { createLensDemo } from "./lensDemoData";

type Tab = "traces" | "investigations";
type WorkspaceProps = { accessToken: string; userRole: string; readOnly: boolean };

export function LensWorkspace(props: WorkspaceProps) {
  const [demoTab, setDemoTab] = useState<Tab | null>(null);
  return demoTab ? (
    <DemoSession initialTab={demoTab} onExit={() => setDemoTab(null)} />
  ) : (
    <LensContent {...props} onDemo={setDemoTab} />
  );
}

function DemoSession({ initialTab, onExit }: { initialTab: Tab; onExit: () => void }) {
  const [demo] = useState(createLensDemo);
  const [client] = useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } }),
  );
  useEffect(
    () => () => {
      client.clear();
    },
    [client],
  );
  return (
    <LensDemoContext.Provider value={demo}>
      <QueryClientProvider client={client}>
        <LensContent accessToken="lens-demo" userRole="" readOnly initialTab={initialTab} onExit={onExit} />
      </QueryClientProvider>
    </LensDemoContext.Provider>
  );
}

function LensContent({
  accessToken,
  userRole,
  readOnly,
  initialTab = "traces",
  onDemo,
  onExit,
}: WorkspaceProps & { initialTab?: Tab; onDemo?: (tab: Tab) => void; onExit?: () => void }) {
  const demo = useLensDemo();
  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringLiteral(["traces", "investigations"]).withOptions({ history: "push" }),
  );
  const [lensId] = useQueryState("lens", parseAsString);
  const [demoTab, setDemoTab] = useState(initialTab);
  const defaultTab = lensId ? "investigations" : "traces";
  const activeTab = demo ? demoTab : tab ?? defaultTab;
  const openDemo = onDemo ? () => onDemo(activeTab) : undefined;
  return (
    <main className="flex w-full min-w-0 flex-1 flex-col gap-5 p-6 md:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
          <Aperture aria-hidden="true" className="size-7" strokeWidth={1.75} />
          Lens
        </h1>
        {demo ? (
          <div className="flex items-center gap-3">
            <span role="status" className="text-sm text-muted-foreground">
              Demo data
            </span>
            <Button variant="outline" size="sm" onClick={onExit}>
              Exit demo
            </Button>
          </div>
        ) : (
          <Button variant="outline" size="sm" onClick={openDemo}>
            Try demo
          </Button>
        )}
      </div>
      <Tabs
        value={activeTab}
        onValueChange={(value) => (demo ? setDemoTab(value as Tab) : void setTab(value as Tab))}
        className="min-h-0 flex-1 gap-4"
      >
        <TabsList variant="line" aria-label="Lens" className="w-full justify-start gap-6 border-b px-0">
          <TabsTrigger value="traces" className="flex-none px-0">
            Traces
          </TabsTrigger>
          <TabsTrigger value="investigations" className="flex-none px-0">
            Investigations
          </TabsTrigger>
        </TabsList>
        <TabsContent value="traces" keepMounted className="min-h-0">
          <AgentTracesPage
            accessToken={accessToken}
            isActive={activeTab === "traces"}
            readOnly={readOnly}
            canMintTracingKey={!demo && isProxyAdminRole(userRole)}
          />
        </TabsContent>
        <TabsContent value="investigations" keepMounted={!!demo}>
          {demo || isProxyAdminTierRole(userRole) ? (
            <LensView accessToken={accessToken} readOnly={readOnly || !isProxyAdminRole(userRole)} onDemo={openDemo} />
          ) : (
            <p className="py-6 text-sm text-muted-foreground">
              Investigations require proxy administrator access. You can still view your traces.
            </p>
          )}
        </TabsContent>
      </Tabs>
    </main>
  );
}
