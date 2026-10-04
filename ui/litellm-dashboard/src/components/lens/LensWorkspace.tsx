"use client";

import { useState, type ReactNode } from "react";
import { Aperture } from "lucide-react";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { DemoNotice } from "@/components/shared/DemoNotice";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LensServicesProvider } from "./LensServicesProvider";
import { LensPreviewContext } from "./LensPreviewButton";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { InvestigationsView } from "./investigations/InvestigationsView";
import { createLensDemo } from "./demo/createLensDemo";
import { LENS_TABS, MemoryLensRoute, useLensRoute, type LensTab } from "./route";

type WorkspaceProps = { accessToken: string; userRole: string; readOnly: boolean };

export function LensWorkspace(props: WorkspaceProps) {
  const [sampleTab, setSampleTab] = useState<LensTab | null>(null);
  return sampleTab ? (
    <SampleSession initialTab={sampleTab} onExit={() => setSampleTab(null)} />
  ) : (
    <LensContent {...props} onPreview={setSampleTab} />
  );
}

function SampleSession({ initialTab, onExit }: { initialTab: LensTab; onExit: () => void }) {
  const [services] = useState(() => createLensDemo());
  return (
    <LensServicesProvider services={services}>
      <MemoryLensRoute initialTab={initialTab}>
        <LensContent
          accessToken="lens-demo"
          userRole="proxy_admin_viewer"
          readOnly
          notice={<DemoNotice onExit={onExit} />}
        />
      </MemoryLensRoute>
    </LensServicesProvider>
  );
}

function LensContent({
  accessToken,
  userRole,
  readOnly,
  notice,
  onPreview,
}: WorkspaceProps & { notice?: ReactNode; onPreview?: (tab: LensTab) => void }) {
  const { tab, lensId, setTab } = useLensRoute();
  const [previewTarget, setPreviewTarget] = useState<HTMLDivElement | null>(null);
  const activeTab = tab ?? (lensId ? "findings" : "traces");
  const preview = (view: LensTab) => ({
    target: previewTarget,
    open: onPreview && activeTab === view ? () => onPreview(view) : undefined,
  });
  return (
    <main className="flex min-h-full w-full min-w-0 flex-1 flex-col gap-2 px-3 pt-2 pb-3">
      {notice}
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
          <div ref={setPreviewTarget} />
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
