"use client";

import { parseAsStringLiteral, useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { extractProxyErrorMessage } from "@/lib/http/client";
import AppDetailPage from "../_components/AppDetailPage";
import AppSettingsForm from "../_components/AppSettingsForm";
import AppBudgetDefaults from "../_components/AppBudgetDefaults";
import { grantedApplicationRows } from "../_components/applications";
import { useOverallAppBudgetGroups } from "../_components/useAppBudgetGroups";
import type { App } from "../_components/appForms";
import CustomerBudgetCatalog, { budgetGroupCatalogs } from "../../customers/_components/CustomerBudgetCatalog";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import { customerUsageDateParsers } from "../../customers/_components/customerUsageRange";

const sections = ["budget", "members"] as const;
const settingsParsers = {
  ...customerAppParsers,
  ...customerUsageDateParsers,
  section: parseAsStringLiteral(sections).withDefault("budget").withOptions({ history: "push" }),
};
const memberCatalog = budgetGroupCatalogs.find(({ scope }) => scope === "app-member")!;

export default function AppSettingsPage() {
  return <AppDetailPage active="settings">{(app) => <AppSettings key={app.app_id} app={app} />}</AppDetailPage>;
}

function AppSettings({ app }: { app: App }) {
  const { t } = useTranslation();
  const [params, setParams] = useQueryStates(settingsParsers);
  const groups = useOverallAppBudgetGroups(params.section === "budget");
  const row = grantedApplicationRows([app])[0];
  return (
    <Tabs
      value={params.section}
      onValueChange={(value) => {
        const section = sections.find((section) => section === value);
        if (section) void setParams({ section });
      }}
      className="gap-6"
    >
      <TabsList aria-label={t("apps.settings.sections")}>
        {sections.map((section) => (
          <TabsTrigger key={section} value={section}>
            {t(`apps.settings.${section}`)}
          </TabsTrigger>
        ))}
      </TabsList>
      <TabsContent value="budget">
        {groups.isPending && <p role="status">{t("apps.loading")}</p>}
        {groups.isError && (
          <ErrorState message={extractProxyErrorMessage(groups.error)} onRetry={() => void groups.refetch()} />
        )}
        {groups.data && row && (
          <AppSettingsForm app={app} resource={row.resource} profiles={groups.data.default_budgets} />
        )}
      </TabsContent>
      <TabsContent value="members" className="w-full space-y-6">
        <p className="text-sm text-muted-foreground">{t("apps.settings.sharedGroupsHint")}</p>
        <CustomerBudgetCatalog catalog={memberCatalog} includeAppDefaults={false} />
        <div className="border-t pt-6">
          <AppBudgetDefaults scope="member" app={app} />
        </div>
      </TabsContent>
    </Tabs>
  );
}
