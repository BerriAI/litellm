"use client";

import { useQueryStates } from "nuqs";
import { Page } from "@/components/shared/Page";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";
import IssueDetail from "@/components/Issues/IssueDetail";
import IssuesInbox from "@/components/Issues/IssuesInbox";
import { issueRouteParsers } from "@/components/Issues/issueRoute";

export default function AlertsPage() {
  const [params] = useQueryStates(issueRouteParsers);
  if (params.issue)
    return (
      <Page className="gap-4">
        <IssueDetail id={params.issue} />
      </Page>
    );
  if (params.model && params.type === "model-deprecations")
    return (
      <Page className="gap-4">
        <ModelDeprecationsPanel />
      </Page>
    );
  return (
    <Page className="h-full gap-4">
      <h1 className="text-xl font-semibold">Issues</h1>
      <IssuesInbox />
    </Page>
  );
}
