"use client";

import { useId, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Tabs as TabsPrimitive } from "@base-ui/react/tabs";
import { Activity, Aperture, ArrowUpRight, ScanSearch } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { LensServicesProvider } from "./LensServicesProvider";
import { LensPreviewContext } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { createLensDemo } from "./demo/createLensDemo";
import { lensQueries } from "./api/queries";
import { useLensApi } from "./services";
import { investigationActivity, type InvestigationActivity } from "./model/status";
import { cn } from "@/lib/cva.config";
import { LENS_TABS, useLensRoute, type LensTab } from "./route";

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
    <div className={cn("flex items-center gap-2 text-xs", demo ? "font-medium text-info" : "text-muted-foreground")}>
      <label htmlFor={id}>Demo data</label>
      <Switch id={id} size="sm" checked={demo} onCheckedChange={onChange} className="data-checked:bg-info" />
    </div>
  );
}

const MODE_ICONS = { traces: Activity, investigations: ScanSearch } as const;

const ACTIVITY_DOT: Record<Exclude<InvestigationActivity, "idle">, { className: string; label: string }> = {
  running: { className: "bg-info motion-safe:animate-pulse", label: "An investigation is running" },
  queued: { className: "bg-muted-foreground/60", label: "An investigation is queued" },
};

function ActivityDot({ activity }: { activity: InvestigationActivity }) {
  if (activity === "idle") return null;
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-1.5 rounded-full animate-in fade-in-0 zoom-in-50 duration-300",
        ACTIVITY_DOT[activity].className,
      )}
    />
  );
}

const FRAME = {
  live: {
    card: "border border-foreground/15",
    tab: "-mb-px border-x border-t border-foreground/15",
    corner: { left: "border-r border-b", right: "border-l border-b", tone: "border-foreground/15" },
  },
  demo: {
    card: "border-2 border-info",
    tab: "-mb-0.5 border-x-2 border-t-2 border-info",
    corner: { left: "border-r-2 border-b-2", right: "border-l-2 border-b-2", tone: "border-info" },
  },
} as const;

const frameOf = (demo: boolean) => FRAME[demo ? "demo" : "live"];

/** Inverted corner joining the tab's side border to the card's top border; plain CSS borders so both snap to the same pixels. */
function NotchCorner({ side, demo }: { side: "left" | "right"; demo: boolean }) {
  const { corner } = frameOf(demo);
  return (
    <span
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute bottom-0 size-3 overflow-hidden",
        side === "left" ? "-left-3" : "-right-3",
      )}
    >
      <span
        className={cn(
          "block size-full shadow-[0_0_0_12px_var(--card)]",
          side === "left" ? "rounded-br-xl" : "rounded-bl-xl",
          corner[side],
          corner.tone,
        )}
      />
    </span>
  );
}

function LensModeSwitch({ activity, demo }: { activity: InvestigationActivity; demo: boolean }) {
  return (
    <div className={cn("relative z-raised rounded-t-2xl bg-card px-1.5 pt-1.5 pb-[7px]", frameOf(demo).tab)}>
      <NotchCorner side="left" demo={demo} />
      <NotchCorner side="right" demo={demo} />
      <TabsPrimitive.List
        aria-label="Lens"
        className="relative inline-flex h-9 items-center rounded-full bg-muted/70 p-1"
      >
        <TabsPrimitive.Indicator className="absolute top-1 bottom-1 left-(--active-tab-left) w-(--active-tab-width) rounded-full bg-background shadow-sm ring-1 ring-border transition-[left,width] duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none" />
        {Object.entries(LENS_TABS).map(([view, label]) => {
          const Icon = MODE_ICONS[view as LensTab];
          return (
            <TabsPrimitive.Tab
              key={view}
              value={view}
              aria-description={
                view === "investigations" && activity !== "idle" ? ACTIVITY_DOT[activity].label : undefined
              }
              className="relative z-raised inline-flex h-full items-center gap-2 rounded-full px-4 text-sm font-medium text-muted-foreground outline-none transition-colors duration-200 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 data-active:text-foreground"
            >
              <Icon aria-hidden="true" className="size-4" />
              {label}
              {view === "investigations" && <ActivityDot activity={activity} />}
            </TabsPrimitive.Tab>
          );
        })}
      </TabsPrimitive.List>
    </div>
  );
}

function useInvestigationActivity(accessToken: string, enabled: boolean): InvestigationActivity {
  const api = useLensApi(accessToken);
  const { data } = useQuery({ ...lensQueries.list(api, false), enabled });
  return investigationActivity(data?.lenses ?? []);
}

const PANEL =
  "flex min-h-0 flex-1 flex-col overflow-y-auto animate-in fade-in-0 duration-300 motion-reduce:animate-none";

function LensContent({ accessToken, userRole, readOnly }: WorkspaceProps) {
  const { tab, lensId, demo, setTab, setDemo } = useLensRoute();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "investigations" : "traces");
  const canInvestigate = isProxyAdminTierRole(userRole);
  const activity = useInvestigationActivity(accessToken, canInvestigate);
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
          <LensModeSwitch activity={activity} demo={demo} />
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
        </div>
      </Tabs>
    </main>
  );
}
