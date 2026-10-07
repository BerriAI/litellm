"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Aperture, ArrowUpRight, Loader2 } from "lucide-react";
import AgentTracesPage from "@/components/lens/traces/list/AgentTracesPage";
import { Button } from "@/components/ui/button";
import type { TraceSummary } from "@/components/lens/traces/types";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { LensServicesProvider, useLensAccessToken, useLensApi, useLiveLensServices } from "./data/LensServices";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { DatasetsView } from "./datasets/DatasetsView";
import { LensSettings } from "./settings/LensSettings";
import { createLensDemo } from "./data/demo/createLensDemo";
import { lensQueries } from "./data/queries";
import { LensModeSwitch } from "./LensModeSwitch";
import { FindingsView } from "./investigations/FindingsView";
import { investigationActivity, listPollInterval } from "./model/status";
import { cn } from "@/lib/cva.config";
import { useDialogRoute, useIssueRoute, useLensRoute, type LensDialog, type LensTab } from "./route";
import { LensGettingStarted } from "./onboarding/LensGettingStarted";
import { useLensReadiness, type LensReadiness } from "./hooks/useLensReadiness";
import { OnboardingProvider, type Onboarding } from "./onboarding/OnboardingContext";
import { traceRefOf, useOpenTraceRouting, type TraceRef } from "@/components/lens/traces/routing";

type WorkspaceProps = { accessToken: string; userRole: string; readOnly: boolean };

export function LensWorkspace(props: WorkspaceProps) {
  const { demo } = useLensRoute();
  return demo ? <SampleSession /> : <LiveSession {...props} />;
}

function LiveSession(props: WorkspaceProps) {
  const services = useLiveLensServices(props.accessToken);
  return (
    <LensServicesProvider services={services}>
      <LensContent userRole={props.userRole} readOnly={props.readOnly} />
    </LensServicesProvider>
  );
}

function SampleSession() {
  const [services] = useState(() => createLensDemo());
  return (
    <LensServicesProvider services={services}>
      <LensContent userRole="proxy_admin_viewer" readOnly />
    </LensServicesProvider>
  );
}

function DemoToggle({ demo, onChange }: { demo: boolean; onChange: (demo: boolean) => void }) {
  const id = useId();
  return (
    <div
      className={cn(
        "flex items-center gap-2 whitespace-nowrap text-xs",
        demo ? "font-medium text-info" : "text-muted-foreground",
      )}
    >
      <label htmlFor={id}>Demo data</label>
      <Switch id={id} size="sm" checked={demo} onCheckedChange={onChange} className="data-checked:bg-info" />
    </div>
  );
}

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

function LensContent({ userRole, readOnly }: Omit<WorkspaceProps, "accessToken">) {
  const accessToken = useLensAccessToken();
  const { tab, lensId, demo, settingUp, setTab, setDemo, setSetup } = useLensRoute();
  const { dialog, openDialog } = useDialogRoute();
  const { issueKey } = useIssueRoute();
  const { trace, openTrace } = useOpenTraceRouting();
  const canViewInvestigations = isProxyAdminTierRole(userRole);
  const isAdmin = isProxyAdminRole(userRole);
  const canConfigure = canViewInvestigations && !readOnly;
  const defaultTab = lensId ? "investigations" : "traces";
  const activeTab = tab === "settings" && !canConfigure ? defaultTab : tab ?? defaultTab;
  const setupState = useLensReadiness(canViewInvestigations);
  const setupLocation = {
    tab: activeTab,
    requested: settingUp,
    canViewInvestigations,
    trace,
    lensId,
    dialog,
    issueKey,
  };
  const showSetup = !demo && needsSetup(setupState, setupLocation);
  const { activity, list } = useLensOverview(
    canViewInvestigations,
    (canConfigure && activeTab === "settings") || settingUp,
  );
  const workers = canConfigure && list ? list.workers : null;
  const leaveSetup = () => {
    setSetup(false);
  };
  const startSetup = () => {
    setSetup(true);
  };
  const exitSetup = (to: LensTab) => {
    leaveSetup();
    setTab(to);
  };
  const showSettings = () => {
    leaveSetup();
    setTab("settings");
  };
  const startFirstInvestigation = () => {
    leaveSetup();
    setTab("investigations");
    openDialog("new");
  };
  const showSentTrace = (trace: TraceSummary) => {
    leaveSetup();
    setTab("traces");
    openTrace(traceRefOf(trace));
  };
  const toggleDemo = (next: boolean) => {
    leaveSetup();
    setDemo(next);
  };
  const onboarding: Onboarding = {
    readOnly,
    canViewInvestigations,
    canInvestigate: isAdmin,
    canMintTracingKey: isAdmin,
    connect: showSettings,
    create: startFirstInvestigation,
    openTrace: showSentTrace,
  };
  return (
    <OnboardingProvider value={onboarding}>
      <main className="flex h-full w-full min-w-0 flex-1 flex-col px-3 pt-3 pb-3 sm:px-4 sm:pb-4">
        <Tabs
          value={activeTab}
          onValueChange={(value) => {
            if (value === "settings") leaveSetup();
            setTab(value as LensTab);
          }}
          className="@container/lens-frame min-h-0 flex-1 gap-0"
        >
          <header className="grid shrink-0 grid-cols-[1fr_auto] items-center gap-x-4 gap-y-2 border-b pb-2 @min-[42rem]/lens-frame:grid-cols-[auto_1fr_auto]">
            <h1 className="flex items-center gap-1.5 text-sm font-semibold">
              <Aperture aria-hidden="true" className="size-4" strokeWidth={2} />
              Lens
            </h1>
            <div className="col-span-2 row-start-2 min-w-0 @min-[42rem]/lens-frame:col-span-1 @min-[42rem]/lens-frame:col-start-2 @min-[42rem]/lens-frame:row-start-1">
              <LensModeSwitch activity={activity} workers={workers} />
            </div>
            <div className="col-start-2 row-start-1 flex items-center justify-end gap-4 @min-[42rem]/lens-frame:col-start-3">
              <a
                href="https://docs.litellm.ai/docs/proxy/lens"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-0.5 text-xs text-muted-foreground hover:text-foreground"
              >
                Docs
                <ArrowUpRight aria-hidden="true" className="size-3" />
              </a>
              <DemoToggle demo={demo} onChange={toggleDemo} />
            </div>
          </header>
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-card">
            {showSetup ? (
              <TabsContent value={activeTab} keepMounted className={cn(PANEL, "p-3 sm:p-5")}>
                {setupState.loading ? (
                  <p role="status" className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
                    <Loader2 aria-hidden="true" className="size-4 animate-spin" />
                    Checking Lens setup…
                  </p>
                ) : (
                  <LensGettingStarted state={setupState} onStart={startSetup} onExit={exitSetup} />
                )}
              </TabsContent>
            ) : (
              <>
                <TabsContent value="traces" keepMounted className={PANEL}>
                  <AgentTracesPage
                    accessToken={accessToken}
                    isActive={activeTab === "traces"}
                    readOnly={readOnly}
                    canMintTracingKey={isAdmin}
                    canViewFindings={canViewInvestigations}
                    onSetUpSignals={canConfigure ? showSettings : undefined}
                  />
                </TabsContent>
                <TabsContent value="findings" className={PANEL}>
                  {canViewInvestigations ? (
                    <FindingsView readOnly={readOnly || !isAdmin} />
                  ) : (
                    <p className="py-6 text-sm text-muted-foreground">Findings require proxy administrator access.</p>
                  )}
                </TabsContent>
                <TabsContent value="investigations" className={PANEL}>
                  {canViewInvestigations ? (
                    <InvestigationsView readOnly={readOnly || !isAdmin} />
                  ) : (
                    <p className="py-6 text-sm text-muted-foreground">
                      Investigations require proxy administrator access. You can still view your traces.
                    </p>
                  )}
                </TabsContent>
                <TabsContent value="datasets" className={PANEL}>
                  <DatasetsPanel canView={canViewInvestigations} isAdmin={isAdmin} readOnly={readOnly} />
                </TabsContent>
              </>
            )}
            {workers && list && (
              <TabsContent value="settings" keepMounted className={cn(PANEL, "p-3 sm:p-5")}>
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
        </Tabs>
      </main>
    </OnboardingProvider>
  );
}

function DatasetsPanel({ canView, isAdmin, readOnly }: { canView: boolean; isAdmin: boolean; readOnly: boolean }) {
  if (!canView)
    return <p className="py-6 text-sm text-muted-foreground">Datasets require proxy administrator access.</p>;
  return <DatasetsView readOnly={readOnly || !isAdmin} />;
}

function needsSetup(
  state: LensReadiness,
  location: {
    tab: LensTab;
    requested: boolean;
    canViewInvestigations: boolean;
    trace: TraceRef | null;
    lensId: string | null;
    dialog: LensDialog | null;
    issueKey: string | null;
  },
) {
  if (location.tab === "settings" || location.tab === "datasets") return false;
  if (location.requested) return true;
  const selected = location.tab === "traces" ? location.trace : location.lensId || location.dialog || location.issueKey;
  if (!state.missingTraces || selected) return false;
  if (location.tab === "traces") return true;
  return location.canViewInvestigations && !state.hasInvestigations && !state.hasRecordedActivity;
}
