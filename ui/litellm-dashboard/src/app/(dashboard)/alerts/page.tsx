"use client";

import { Page, PageTabs, PageTabsList, PageTabsTrigger, PageTabsContent } from "@/components/shared/Page";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";

export default function AlertsPage() {
  return (
    <Page className="h-full gap-4">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Alerts</h1>
        <p className="mt-1 text-sm text-muted-foreground">Review and resolve issues across your keys and teams.</p>
      </header>
      <PageTabs defaultValue="model-deprecations" className="gap-4">
        <PageTabsList>
          <PageTabsTrigger value="model-deprecations">Model deprecations</PageTabsTrigger>
        </PageTabsList>
        <PageTabsContent value="model-deprecations">
          <ModelDeprecationsPanel />
        </PageTabsContent>
      </PageTabs>
    </Page>
  );
}
