"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Aperture, ArrowUpRight, Loader2 } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Button } from "@/components/ui/button";
import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { LensServicesProvider, useLensApi, useLiveLensServices } from "./LensServices";
import { LensPreviewContext } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { LensSettings } from "./settings/LensSettings";
import { createLensDemo } from "./demo/createLensDemo";
import { lensQueries } from "./api/queries";
import { frameOf, LensModeSwitch } from "./LensModeSwitch";
import { investigationActivity, listPollInterval } from "./model/status";
import { cn } from "@/lib/cva.config";
import { useDialogRoute, useLensRoute, type LensDialog, type LensTab } from "./route";
import { LensGettingStarted } from "./setup/LensGettingStarted";
import { useLensSetup, type LensSetupState } from "./setup/useLensSetup";
import { traceRefOf, useOpenTraceRouting } from "@/components/view_logs/TraceView/traceRouting";

type WorkspaceProps = { accessToken: string; userRole: string; readOnly: boolean };

export function LensWorkspace(props: WorkspaceProps) {
  const { demo } = useLensRoute();
  return demo ? <SampleSession /> : <LiveSession {...props} />;
}

function LiveSession(props: WorkspaceProps) {
  const services = useLiveLensServices(props.accessToken);
  return (
    <LensServicesProvider services={services}>
      <LensContent {...props} />
    </LensServicesProvider>
  );
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
    <div className={cn("flex items-center gap-2 text-xs", demo ? "font-medium text-info" : "text-muted-foreground")}>
      <label htmlFor={id}>Demo data</label>
      <Switch id={id} size="sm" checked={demo} onCheckedChange={onChange} className="data-checked:bg-info" />
    </div>
  );
}

/** The inline investigation editor marks the tab so the notch says where you are, not just which tab is open. */
const SETUP_LABELS: Partial<Record<LensDialog, string>> = { new: "New", edit: "Editing", duplicate: "Duplicate" };

/** The one always-mounted `/lens` observer; every other reader is a plain cache subscriber. */
function useLensOverview(enabled: boolean, settingsOpen: boolean) {
  const api = useLensApi();
  const { data } = useQuery({
    ...lensQueries.list(api),
    enabled,
    refetchInterval: (query) => listPollInterval(query.state.data, settingsOpen, Date.now()),
  });
  return { activity: investigationActivity(data?.lenses ?? []), list: data };
}

const PANEL =
  "flex min-h-0 flex-1 flex-col overflow-y-auto animate-in fade-in-0 duration-300 motion-reduce:animate-none";

function LensContent({ accessToken, userRole, readOnly }: WorkspaceProps) {
  const { tab, lensId, demo, settingUp, setTab, setDemo, setSetup } = useLensRoute();
  const { dialog, openDialog } = useDialogRoute();
  const { openTrace } = useOpenTraceRouting();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "investigations" : "traces");
  const canInvestigate = isProxyAdminTierRole(userRole);
  const canConfigure = canInvestigate && !readOnly;
  const setupState = useLensSetup(accessToken, !demo, canInvestigate);
  const setupLocation = { tab: activeTab, canInvestigate, selected: !!lensId, requested: settingUp };
  const showSetup = !demo && needsSetup(setupState, setupLocation);
  const showSetupButton = !demo && !showSetup && isProxyAdminRole(userRole);
  const { activity, list } = useLensOverview(canInvestigate, (canConfigure && activeTab === "settings") || showSetup);
  const workers = canConfigure && list ? list.workers : null;
  const startSetup = () => {
    if (!settingUp) setSetup(true);
  };
  const showTraces = () => {
    setSetup(false);
    setTab(setupState.tracesReady ? "traces" : "investigations");
  };
  const showSettings = () => {
    setSetup(false);
    setTab("settings");
  };
  const startFirstInvestigation = () => {
    setSetup(false);
    setTab("investigations");
    openDialog("new");
  };
  const showSentTrace = (trace: TraceSummary) => {
    setSetup(false);
    setTab("traces");
    openTrace(traceRefOf(trace));
  };
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: !demo && !showSetup && activeTab === view ? () => setDemo(true) : undefined,
  });
  const setupContent = setupState.loading ? (
    <p role="status" className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
      <Loader2 aria-hidden="true" className="size-4 animate-spin" />
      Checking Lens setup…
    </p>
  ) : (
    <div className={cn(PANEL, "p-4")}>
      <LensGettingStarted
        accessToken={accessToken}
        state={setupState}
        readOnly={readOnly}
        canInvestigate={isProxyAdminRole(userRole)}
        canMintTracingKey={isProxyAdminRole(userRole)}
        onStart={startSetup}
        onExit={showTraces}
        onConnect={showSettings}
        onCreate={startFirstInvestigation}
        onTrace={showSentTrace}
        onDemo={() => setDemo(true)}
      />
    </div>
  );
  return (
    <main className="flex h-full w-full min-w-0 flex-1 flex-col px-4 pt-3 pb-4">
      <Tabs value={activeTab} onValueChange={(value) => setTab(value as LensTab)} className="min-h-0 flex-1 gap-0">
        <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-3">
          <div className="flex min-w-0 flex-col gap-1 pb-3">
            <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight">
              <Aperture aria-hidden="true" className="size-5" strokeWidth={2} />
              Lens
            </h1>
            <p className="truncate text-xs text-muted-foreground">
              Trace your agents and investigate what goes wrong.{" "}
              <a
                href="https://docs.litellm.ai/docs/proxy/lens"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-0.5 font-medium text-foreground/80 underline-offset-2 hover:text-foreground hover:underline"
              >
                Docs
                <ArrowUpRight aria-hidden="true" className="size-3" />
              </a>
            </p>
          </div>
          <LensModeSwitch
            activity={activity}
            demo={demo}
            workers={workers}
            setup={activeTab === "investigations" && dialog ? SETUP_LABELS[dialog] : undefined}
          />
          <div className="flex min-w-0 flex-wrap items-center justify-end gap-3 pb-3">
            {showSetupButton && (
              <Button variant="outline" size="sm" onClick={startSetup}>
                Set up Lens
              </Button>
            )}
            <div ref={setPreviewTarget} />
            <DemoToggle demo={demo} onChange={setDemo} />
          </div>
        </div>
        <div className={cn("flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl bg-card", frameOf(demo).card)}>
          {showSetup && setupContent}
          <div hidden={showSetup} className="contents">
            <TabsContent value="traces" keepMounted className={PANEL}>
              <LensPreviewContext.Provider value={preview("traces")}>
                <AgentTracesPage
                  accessToken={accessToken}
                  isActive={activeTab === "traces" && !showSetup}
                  readOnly={readOnly}
                  canMintTracingKey={isProxyAdminRole(userRole)}
                />
              </LensPreviewContext.Provider>
            </TabsContent>
            <TabsContent value="investigations" className={cn(PANEL, "p-4")}>
              <LensPreviewContext.Provider value={preview("investigations")}>
                {canInvestigate ? (
                  <InvestigationsView accessToken={accessToken} readOnly={readOnly || !isProxyAdminRole(userRole)} />
                ) : (
                  <p className="py-6 text-sm text-muted-foreground">
                    Investigations require proxy administrator access. You can still view your traces.
                  </p>
                )}
              </LensPreviewContext.Provider>
            </TabsContent>
            {workers && list && (
              <TabsContent value="settings" keepMounted className={cn(PANEL, "p-6")}>
                <LensSettings
                  list={list}
                  workerReadyAction={
                    list.lenses.length === 0 ? (
                      <Button className="w-full" onClick={startFirstInvestigation}>
                        New investigation
                      </Button>
                    ) : undefined
                  }
                  onOpenTraces={() => setTab("traces")}
                />
              </TabsContent>
            )}
          </div>
        </div>
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
  if (tab === "settings" || !state.missingTraces) return false;
  if (tab === "traces") return true;
  const hasActivity = state.hasInvestigations || state.hasRequests || selected;
  return canInvestigate && !hasActivity;
}
