"use client";

import { useId, useState } from "react";
import { Aperture, Info } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LensServicesProvider } from "./LensServicesProvider";
import { LensPreviewContext } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { createLensDemo } from "./demo/createLensDemo";
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

function LensContent({ accessToken, userRole, readOnly }: WorkspaceProps) {
  const { tab, lensId, demo, setTab, setDemo } = useLensRoute();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "findings" : "traces");
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: !demo && activeTab === view ? () => setDemo(true) : undefined,
  });
  return (
    <main className="flex h-full w-full min-w-0 flex-1 flex-col gap-2 px-3 pt-2 pb-3">
      <Tabs value={activeTab} onValueChange={(value) => setTab(value as LensTab)} className="min-h-0 flex-1 gap-2">
        <div className="flex min-h-8 flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <h1 className="flex items-center gap-1.5 text-sm font-semibold tracking-tight">
              <Aperture aria-hidden="true" className="size-4" strokeWidth={2} />
              Lens
            </h1>
            <TabsList aria-label="Lens" className="h-8">
              {Object.entries(LENS_TABS).map(([view, label]) => (
                <TabsTrigger key={view} value={view} className="px-3">
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
          <div className="flex items-center gap-3">
            <div ref={setPreviewTarget} />
            <DemoToggle demo={demo} onChange={setDemo} />
          </div>
        </div>
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
        {(["findings", "investigations"] as const).map((view) => (
          <TabsContent key={view} value={view} className="flex min-h-0 flex-col overflow-y-auto">
            <LensPreviewContext.Provider value={preview(view)}>
              {isProxyAdminTierRole(userRole) ? (
                <InvestigationsView
                  view={view}
                  active={activeTab === view}
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
        ))}
      </Tabs>
    </main>
  );
}
