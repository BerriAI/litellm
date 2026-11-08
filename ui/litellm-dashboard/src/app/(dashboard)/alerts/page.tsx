"use client";

import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { Page } from "@/components/shared/Page";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";
import IssueDetail from "@/components/Issues/IssueDetail";
import IssuesInbox from "@/components/Issues/IssuesInbox";
import { issueRouteParsers } from "@/components/Issues/issueRoute";

export default function AlertsPage() {
  const { t } = useTranslation();
  const [params] = useQueryStates(issueRouteParsers);
  if (params.issue)
    return (
      <Page className="mx-auto max-w-[1440px] gap-4">
        <IssueDetail id={params.issue} />
      </Page>
    );
  if (params.model)
    return (
      <Page className="mx-auto max-w-[1440px] gap-4">
        <ModelDeprecationsPanel />
      </Page>
    );
  return (
    <Page className="mx-auto h-full max-w-[1440px] gap-4">
      <h1 className="text-xl font-semibold">{t("shell.nav.items.alerts")}</h1>
      <IssuesInbox />
    </Page>
  );
}
