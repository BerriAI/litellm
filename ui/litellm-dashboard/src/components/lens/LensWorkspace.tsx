"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Aperture, ArrowUpRight } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Button } from "@/components/ui/button";
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
  const { tab, lensId, demo, setTab, setDemo } = useLensRoute();
  const { dialog, openDialog } = useDialogRoute();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "investigations" : "traces");
  const canInvestigate = isProxyAdminTierRole(userRole);
  const canConfigure = canInvestigate && !readOnly;
  const { activity, list } = useLensOverview(canInvestigate, canConfigure && activeTab === "settings");
  const workers = canConfigure && list ? list.workers : null;
  const startFirstInvestigation = () => {
    setTab("investigations");
    openDialog("new");
  };
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: !demo && activeTab === view ? () => setDemo(true) : undefined,
  });
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
            <div ref={setPreviewTarget} />
            <DemoToggle demo={demo} onChange={setDemo} />
          </div>
        </div>
        <div className={cn("flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl bg-card", frameOf(demo).card)}>
          <TabsContent value="traces" keepMounted className={PANEL}>
            <LensPreviewContext.Provider value={preview("traces")}>
              <AgentTracesPage
                accessToken={accessToken}
                isActive={activeTab === "traces"}
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
      </Tabs>
    </main>
  );
}
