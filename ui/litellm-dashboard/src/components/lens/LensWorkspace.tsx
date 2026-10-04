"use client";

import { useId, useState } from "react";
import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { Activity, Aperture, Info, Loader2, ScanSearch } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { LensServicesProvider } from "./LensServicesProvider";
import { LensPreviewContext } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { createLensDemo } from "./demo/createLensDemo";
import { LENS_TABS, useLensRoute, type LensTab } from "./route";
import { Button } from "@/components/ui/button";
import { LensGettingStarted } from "./setup/LensGettingStarted";
import { useLensSetup, type LensSetupState } from "./setup/useLensSetup";

type WorkspaceProps = { accessToken: string; userRole: string; readOnly: boolean };

export function LensWorkspace(props: WorkspaceProps) {
  const { demo } = useLensRoute();
  return demo ? <SampleSession /> : <LensContent {...props} />;
}

function SampleSession() {
  const [services] = useState(() => createLensDemo());
  return (
    <LensServicesProvider services={services}>
      <LensContent accessToken="lens-demo" userRole="proxy_admin_viewer" readOnly />
    </LensServicesProvider>
  );
}

function DemoToggle({ demo, onChange }: { demo: boolean; onChange: (demo: boolean) => void }) {
  const id = useId();
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      {demo && (
        <p role="status" className="flex items-center gap-1.5 font-medium text-info">
          <Info aria-hidden="true" className="size-3.5 shrink-0" />
          You’re viewing demo data
        </p>
      )}
      <label htmlFor={id}>Demo data</label>
      <Switch id={id} size="sm" checked={demo} onCheckedChange={onChange} />
    </div>
  );
}

const MODE_ICONS = { traces: Activity, investigations: ScanSearch } as const;

function LensModeSwitch() {
  return (
    <TabsPrimitive.List
      aria-label="Lens"
      className="relative inline-flex h-8 items-center rounded-full border border-border bg-muted/60 p-0.5"
    >
      <TabsPrimitive.Indicator className="absolute top-0.5 bottom-0.5 left-(--active-tab-left) w-(--active-tab-width) rounded-full bg-background shadow-sm ring-1 ring-border transition-[left,width] duration-200 ease-out motion-reduce:transition-none" />
      {Object.entries(LENS_TABS).map(([view, label]) => {
        const Icon = MODE_ICONS[view as LensTab];
        return (
          <TabsPrimitive.Tab
            key={view}
            value={view}
            className="relative z-raised inline-flex h-full items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-active:text-foreground"
          >
            <Icon aria-hidden="true" className="size-3.5" />
            {label}
          </TabsPrimitive.Tab>
        );
      })}
    </TabsPrimitive.List>
  );
}

function LensContent({ accessToken, userRole, readOnly }: WorkspaceProps) {
  const { tab, lensId, demo, settingUp, setTab, setLensId, setDemo, setSetup } = useLensRoute();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "investigations" : "traces");
  const canInvestigate = isProxyAdminTierRole(userRole);
  const setupState = useLensSetup(accessToken, !demo, canInvestigate, settingUp);
  const setupLocation = { tab: activeTab, canInvestigate, selected: !!lensId, requested: settingUp };
  const showSetup = !demo && needsSetup(setupState, setupLocation);
  const showSetupButton = !demo && !showSetup && isProxyAdminRole(userRole);
  const startSetup = () => {
    if (!settingUp) setSetup(true);
  };
  const showTraces = () => {
    setSetup(false);
    setTab(setupState.tracesReady ? "traces" : "investigations");
  };
  const checkingSetup = !demo && setupState.loading;
  const showCreated = (id: string) => {
    setSetup(false);
    setLensId(id);
    setTab("investigations");
  };
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: !demo && activeTab === view ? () => setDemo(true) : undefined,
  });
  const setupContent = checkingSetup ? (
    <p role="status" className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
      <Loader2 aria-hidden="true" className="size-4 animate-spin" />
      Checking Lens setup…
    </p>
  ) : (
    <TabsContent value={activeTab} keepMounted className="min-h-0 overflow-y-auto">
      <LensGettingStarted
        accessToken={accessToken}
        state={setupState}
        readOnly={readOnly}
        canInvestigate={isProxyAdminRole(userRole)}
        canMintTracingKey={isProxyAdminRole(userRole)}
        onStart={startSetup}
        onExit={showTraces}
        onCreated={showCreated}
        onDemo={() => setDemo(true)}
      />
    </TabsContent>
  );
  return (
    <main className="flex h-full w-full min-w-0 flex-1 flex-col gap-2 px-3 pt-2 pb-3">
      <Tabs value={activeTab} onValueChange={(value) => setTab(value as LensTab)} className="min-h-0 flex-1 gap-2">
        <div className="flex min-h-8 flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <h1 className="flex items-center gap-1.5 text-sm font-semibold tracking-tight">
              <Aperture aria-hidden="true" className="size-4" strokeWidth={2} />
              Lens
            </h1>
            <LensModeSwitch />
          </div>
          <div className="flex items-center gap-3">
            {showSetupButton && (
              <Button variant="outline" size="sm" onClick={startSetup}>
                Set up Lens
              </Button>
            )}
            <div ref={setPreviewTarget} />
            <DemoToggle demo={demo} onChange={setDemo} />
          </div>
        </div>
        {checkingSetup || showSetup ? (
          setupContent
        ) : (
          <>
            <TabsContent value="traces" keepMounted className="flex min-h-0 flex-col overflow-y-auto">
              <LensPreviewContext.Provider value={preview("traces")}>
                <AgentTracesPage
                  accessToken={accessToken}
                  isActive={activeTab === "traces"}
                  readOnly={readOnly}
                  canMintTracingKey={isProxyAdminRole(userRole)}
                />
              </LensPreviewContext.Provider>
            </TabsContent>
            <TabsContent value="investigations" className="flex min-h-0 flex-col overflow-y-auto">
              <LensPreviewContext.Provider value={preview("investigations")}>
                {isProxyAdminTierRole(userRole) ? (
                  <InvestigationsView
                    active={activeTab === "investigations"}
                    accessToken={accessToken}
                    readOnly={readOnly || !isProxyAdminRole(userRole)}
                  />
                ) : (
                  <p className="py-6 text-sm text-muted-foreground">
                    Investigations require proxy administrator access. You can still view your traces.
                  </p>
                )}
              </LensPreviewContext.Provider>
            </TabsContent>
          </>
        )}
      </Tabs>
    </main>
  );
}

function needsSetup(
  state: LensSetupState,
  {
    tab,
    canInvestigate,
    selected,
    requested,
  }: { tab: LensTab; canInvestigate: boolean; selected: boolean; requested: boolean },
) {
  if (requested) return true;
  if (!state.missingTraces) return false;
  if (tab === "traces") return true;
  const hasActivity = state.hasInvestigations || state.requestsReady || selected;
  return canInvestigate && !hasActivity;
}
