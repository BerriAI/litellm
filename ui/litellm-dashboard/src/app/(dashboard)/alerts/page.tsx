"use client";

import { Page, PageTabs, PageTabsList, PageTabsTrigger, PageTabsContent } from "@/components/shared/Page";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";
import UsageAlertsPanel from "@/components/UsageAlerts/UsageAlertsPanel";
import { parseAsStringLiteral, useQueryState } from "nuqs";

const ALERT_TYPES = ["model-deprecations", "usage"] as const;

export default function AlertsPage() {
  const [type, setType] = useQueryState("type", parseAsStringLiteral(ALERT_TYPES).withDefault("model-deprecations"));
  return (
    <Page className="h-full gap-4">
      <PageTabs
        value={type}
        onValueChange={(value) => void setType(value as (typeof ALERT_TYPES)[number])}
        className="gap-4"
      >
        <PageTabsList>
          <PageTabsTrigger value="model-deprecations">Model deprecations</PageTabsTrigger>
          <PageTabsTrigger value="usage">Performance</PageTabsTrigger>
        </PageTabsList>
        <PageTabsContent value="model-deprecations">
          <ModelDeprecationsPanel />
        </PageTabsContent>
        <PageTabsContent value="usage">
          <UsageAlertsPanel />
        </PageTabsContent>
      </PageTabs>
    </Page>
  );
}
