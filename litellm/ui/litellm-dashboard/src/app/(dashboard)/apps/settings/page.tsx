"use client";

import { createSerializer, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import AppDetailPage from "../_components/AppDetailPage";
import AppSettingsForm from "../_components/AppSettingsForm";
import AppBudgetDefaults from "../_components/AppBudgetDefaults";
import { grantedApplicationRows } from "../_components/applications";
import { useOverallAppBudgetGroups } from "../_components/useAppBudgetGroups";
import type { App } from "../_components/appForms";
import CustomerBudgetCatalog, { budgetGroupCatalogs } from "../../customers/_components/CustomerBudgetCatalog";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import { customerUsageDateParsers } from "../../customers/_components/customerUsageRange";

const sections = ["budget", "members", "groups"] as const;
const groupScopes = ["overall-app", "app-member"] as const;
const settingsParsers = {
  ...customerAppParsers,
  ...customerUsageDateParsers,
  section: parseAsStringLiteral(sections).withDefault("budget").withOptions({ history: "push" }),
  group_scope: parseAsStringLiteral(groupScopes).withDefault("overall-app"),
};
const settingsHref = createSerializer(settingsParsers);
const appCatalogs = budgetGroupCatalogs.filter(({ scope }) => scope === "overall-app" || scope === "app-member");

export default function AppSettingsPage() {
  return <AppDetailPage active="settings">{(app) => <AppSettings key={app.app_id} app={app} />}</AppDetailPage>;
}

function AppSettings({ app }: { app: App }) {
  const { t } = useTranslation();
  const [params, setParams] = useQueryStates(settingsParsers);
  const groups = useOverallAppBudgetGroups(params.section === "budget");
  const row = grantedApplicationRows([app])[0];
  const catalog = appCatalogs.find(({ scope }) => scope === params.group_scope)!;
  const linkTo = (section: (typeof sections)[number]) => settingsHref(uiHref("apps/settings"), { ...params, section });
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
          <section className="max-w-5xl overflow-hidden rounded-lg border bg-card">
            <AppSettingsForm
              app={app}
              resource={row.resource}
              profiles={groups.data.default_budgets}
              overallGroupsHref={settingsHref(uiHref("apps/settings"), {
                ...params,
                section: "groups",
                group_scope: "overall-app",
              })}
              memberDefaultsHref={linkTo("members")}
            />
          </section>
        )}
      </TabsContent>
      <TabsContent value="members" className="max-w-5xl space-y-4">
        <AppBudgetDefaults scope="member" app={app} />
        <a
          href={settingsHref(uiHref("apps/settings"), { ...params, section: "groups", group_scope: "app-member" })}
          className="inline-block text-sm text-primary hover:underline"
        >
          {t("apps.settings.manageMemberGroups")}
        </a>
      </TabsContent>
      <TabsContent value="groups" className="space-y-5">
        <p className="rounded-lg border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
          {t("apps.settings.sharedGroupsHint")}
        </p>
        <SearchSelect
          aria-label={t("apps.settings.groupScope")}
          className="w-full sm:w-72"
          value={params.group_scope}
          options={appCatalogs.map(({ scope, label }) => ({ value: scope, label: t(label) }))}
          onValueChange={(value) => {
            const group_scope = groupScopes.find((scope) => scope === value);
            if (group_scope) void setParams({ group_scope });
          }}
        />
        <CustomerBudgetCatalog key={catalog.scope} catalog={catalog} includeAppDefaults={false} />
      </TabsContent>
    </Tabs>
  );
}
