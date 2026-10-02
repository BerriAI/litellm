"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Aperture } from "lucide-react";
import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import AgentTracesPage from "@/components/view_logs/TraceView/AgentTracesPage";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { LensView } from "./_components/LensView";

export default function LensPage() {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringLiteral(["traces", "investigations"]).withOptions({ history: "push" }),
  );
  const [lensId] = useQueryState("lens", parseAsString);
  const defaultTab = isProxyAdminTierRole(userRole ?? "") || lensId ? "investigations" : "traces";
  const activeTab = tab ?? defaultTab;
  if (!accessToken) return null;
  return (
    <main className="flex w-full min-w-0 flex-1 flex-col gap-5 p-6 md:p-8">
      <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
        <Aperture aria-hidden="true" className="size-7" strokeWidth={1.75} />
        Lens
      </h1>
      <Tabs
        value={activeTab}
        onValueChange={(value) => void setTab(value as "traces" | "investigations")}
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
            readOnly={isViewOnly}
            canMintTracingKey={isProxyAdminRole(userRole ?? "")}
          />
        </TabsContent>
        <TabsContent value="investigations">
          {isProxyAdminTierRole(userRole ?? "") ? (
            <LensView accessToken={accessToken} readOnly={isViewOnly || !isProxyAdminRole(userRole ?? "")} />
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
