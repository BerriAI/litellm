"use client";

import { useEffect, useState } from "react";
import { Aperture, Loader2 } from "lucide-react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { DemoNotice } from "@/components/shared/DemoNotice";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LensDemoContext, useLensDemo } from "./LensDemoContext";
import { LensServicesContext } from "./services";
import { LensPreviewTarget } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { createLensDemo } from "./demo/createLensDemo";
import { Button } from "@/components/ui/button";
import { LensGettingStarted } from "./setup/LensGettingStarted";
import { useLensSetup, type LensSetupState } from "./setup/useLensSetup";

type Tab = "traces" | "findings" | "investigations";
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
    <LensServicesContext.Provider value={demo.services}>
      <LensDemoContext.Provider value={demo}>
        <QueryClientProvider client={client}>
          <LensContent accessToken="lens-demo" userRole="" readOnly initialTab={initialTab} onExit={onExit} />
        </QueryClientProvider>
      </LensDemoContext.Provider>
    </LensServicesContext.Provider>
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
    parseAsStringLiteral(["traces", "findings", "investigations"]).withOptions({ history: "push" }),
  );
  const [lensId, setLensId] = useQueryState("lens", parseAsString);
  const [setup, setSetup] = useQueryState("setup", parseAsStringLiteral(["lens"]).withOptions({ history: "push" }));
  const [demoTab, setDemoTab] = useState(initialTab);
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const defaultTab = lensId ? "findings" : "traces";
  const activeTab = demo ? demoTab : tab ?? defaultTab;
  const openDemo = onDemo ? () => onDemo(activeTab) : undefined;
  const canInvestigate = isProxyAdminTierRole(userRole);
  const setupState = useLensSetup(accessToken, !demo, canInvestigate, setup === "lens");
  const setupLocation = { tab: activeTab, canInvestigate, selected: !!lensId, requested: setup === "lens" };
  const showSetup = !demo && needsSetup(setupState, setupLocation);
  const showSetupButton = !demo && !showSetup && isProxyAdminRole(userRole);
  const startSetup = () => {
    if (setup !== "lens") void setSetup("lens");
  };
  const showTraces = () => {
    void setSetup(null);
    void setTab(setupState.tracesReady ? "traces" : "investigations");
  };
  const checkingSetup = !demo && setupState.loading;
  const showCreated = (id: string) => {
    void setSetup(null);
    void setLensId(id);
    void setTab("findings");
  };
  const sharedSetup = checkingSetup ? (
    <p role="status" className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
      <Loader2 aria-hidden="true" className="size-4 animate-spin" />
      Checking Lens setup…
    </p>
  ) : (
    <TabsContent value={activeTab} keepMounted className="min-w-0">
      <LensGettingStarted
        accessToken={accessToken}
        state={setupState}
        readOnly={readOnly}
        canInvestigate={isProxyAdminRole(userRole)}
        canMintTracingKey={isProxyAdminRole(userRole)}
        onStart={startSetup}
        onExit={showTraces}
        onCreated={showCreated}
        onDemo={openDemo}
      />
    </TabsContent>
  );
  return (
    <LensPreviewTarget.Provider value={previewTarget}>
      <main className="flex min-h-full w-full min-w-0 flex-1 flex-col gap-2 px-3 pt-2 pb-3">
        {demo && <DemoNotice onExit={onExit} />}
        <Tabs
          value={activeTab}
          onValueChange={(value) => (demo ? setDemoTab(value as Tab) : void setTab(value as Tab))}
          className="min-h-0 flex-1 gap-2"
        >
          <div className="flex min-h-8 flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <h1 className="flex items-center gap-1.5 text-sm font-semibold tracking-tight">
                <Aperture aria-hidden="true" className="size-4" strokeWidth={2} />
                Lens
              </h1>
              <TabsList aria-label="Lens" className="h-8">
                <TabsTrigger value="traces" className="px-3">
                  Traces
                </TabsTrigger>
                <TabsTrigger value="findings" className="px-3">
                  Findings
                </TabsTrigger>
                <TabsTrigger value="investigations" className="px-3">
                  Investigations
                </TabsTrigger>
              </TabsList>
            </div>
            <div className="flex items-center gap-2">
              {showSetupButton && (
                <Button variant="outline" size="sm" onClick={startSetup}>
                  Set up Lens
                </Button>
              )}
              <div ref={setPreviewTarget} />
            </div>
          </div>
          {checkingSetup || showSetup ? (
            sharedSetup
          ) : (
            <>
              <TabsContent value="traces" keepMounted className="flex min-h-0 flex-col">
                <AgentTracesPage
                  accessToken={accessToken}
                  isActive={activeTab === "traces"}
                  readOnly={readOnly}
                  canMintTracingKey={!demo && isProxyAdminRole(userRole)}
                  onDemo={activeTab === "traces" ? openDemo : undefined}
                />
              </TabsContent>
              {(["findings", "investigations"] as const).map((view) => (
                <TabsContent key={view} value={view} keepMounted={!!demo} className="flex min-h-0 flex-col">
                  {demo || isProxyAdminTierRole(userRole) ? (
                    <InvestigationsView
                      view={view}
                      active={activeTab === view}
                      accessToken={accessToken}
                      readOnly={readOnly || !isProxyAdminRole(userRole)}
                      onDemo={activeTab === view ? openDemo : undefined}
                    />
                  ) : (
                    <p className="py-6 text-sm text-muted-foreground">
                      Investigations require proxy administrator access. You can still view your traces.
                    </p>
                  )}
                </TabsContent>
              ))}
            </>
          )}
        </Tabs>
      </main>
    </LensPreviewTarget.Provider>
  );
}

function needsSetup(
  state: LensSetupState,
  {
    tab,
    canInvestigate,
    selected,
    requested,
  }: { tab: Tab; canInvestigate: boolean; selected: boolean; requested: boolean },
) {
  if (requested) return true;
  if (!state.missingTraces) return false;
  if (tab === "traces") return true;
  const hasActivity = state.hasInvestigations || state.requestsReady || selected;
  return canInvestigate && !hasActivity;
}
