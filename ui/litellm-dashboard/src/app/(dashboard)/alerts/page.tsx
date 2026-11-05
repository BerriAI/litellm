"use client";

import { Page, PageTabs, PageTabsList, PageTabsTrigger, PageTabsContent } from "@/components/shared/Page";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";

export default function AlertsPage() {
  return (
    <Page className="h-full gap-4">
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
