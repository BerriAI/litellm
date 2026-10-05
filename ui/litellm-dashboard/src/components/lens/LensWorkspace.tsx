"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Aperture, ArrowUpRight } from "lucide-react";
import AgentTracesPage from "@/components/lens/traces/list/AgentTracesPage";
import { Button } from "@/components/ui/button";
import type { TraceSummary } from "@/components/lens/traces/types";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { LensServicesProvider, useLensAccessToken, useLensApi, useLiveLensServices } from "./data/LensServices";
import { LensPreviewContext } from "@/components/lens/ui/LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { LensSettings } from "./settings/LensSettings";
import { createLensDemo } from "./data/demo/createLensDemo";
import { lensQueries } from "./data/queries";
import { LensModeSwitch } from "./LensModeSwitch";
import { frameCard } from "./ui/frame";
import { investigationActivity, listPollInterval } from "./model/status";
import { cn } from "@/lib/cva.config";
import { useDialogRoute, useLensRoute, type LensDialog, type LensTab } from "./route";
import { LensIntroDialog, useLensIntro } from "./onboarding/LensIntroDialog";
import { OnboardingProvider, type Onboarding } from "./onboarding/OnboardingContext";
import { traceRefOf, useOpenTraceRouting } from "@/components/lens/traces/routing";

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

function LensContent({ userRole, readOnly }: Omit<WorkspaceProps, "accessToken">) {
  const accessToken = useLensAccessToken();
  const { tab, lensId, demo, settingUp, setTab, setDemo, setSetup } = useLensRoute();
  const { dialog, openDialog } = useDialogRoute();
  const { openTrace } = useOpenTraceRouting();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const canViewInvestigations = isProxyAdminTierRole(userRole);
  const isAdmin = isProxyAdminRole(userRole);
  const canConfigure = canViewInvestigations && !readOnly;
  const defaultTab = lensId ? "investigations" : "traces";
  const activeTab = tab === "settings" && !canConfigure ? defaultTab : tab ?? defaultTab;
  const intro = useLensIntro({ demo, settingUp });
  const { activity, list } = useLensOverview(
    canViewInvestigations,
    (canConfigure && activeTab === "settings") || intro.open,
  );
  const workers = canConfigure && list ? list.workers : null;
  const leaveIntro = (forever = false) => {
    if (!intro.open) return;
    intro.close(forever);
    setSetup(false);
  };
  const startSetup = () => {
    if (!settingUp) setSetup(true);
  };
  const exitIntro = (to: LensTab) => {
    leaveIntro();
    setTab(to);
  };
  const showSettings = () => {
    leaveIntro();
    setTab("settings");
  };
  const startFirstInvestigation = () => {
    leaveIntro();
    setTab("investigations");
    openDialog("new");
  };
  const showSentTrace = (trace: TraceSummary) => {
    leaveIntro();
    setTab("traces");
    openTrace(traceRefOf(trace));
  };
  const enterDemo = () => {
    leaveIntro();
    setDemo(true);
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
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: !demo && !intro.open && activeTab === view ? enterDemo : undefined,
  });
  return (
    <OnboardingProvider value={onboarding}>
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
              <div ref={setPreviewTarget} />
              <DemoToggle demo={demo} onChange={(next) => (next ? enterDemo() : setDemo(false))} />
            </div>
          </div>
          <div className={frameCard({ session: demo ? "demo" : "live" })}>
            <TabsContent value="traces" keepMounted className={PANEL}>
              <LensPreviewContext.Provider value={preview("traces")}>
                <AgentTracesPage
                  accessToken={accessToken}
                  isActive={activeTab === "traces"}
                  readOnly={readOnly}
                  canMintTracingKey={isAdmin}
                />
              </LensPreviewContext.Provider>
            </TabsContent>
            <TabsContent value="investigations" className={PANEL}>
              <LensPreviewContext.Provider value={preview("investigations")}>
                {canViewInvestigations ? (
                  <InvestigationsView readOnly={readOnly || !isAdmin} />
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
        </Tabs>
        <LensIntroDialog
          open={intro.open}
          onClose={leaveIntro}
          onStart={startSetup}
          onExit={exitIntro}
          onDemo={enterDemo}
        />
      </main>
    </OnboardingProvider>
  );
}
