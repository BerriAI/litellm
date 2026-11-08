import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import { ArrowLeft, Copy } from "lucide-react";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { getProxyBaseUrl } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button } from "@/components/ui/button";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { activityPresets, entityActivityParsers, localActivityDay } from "@/components/UsagePage/entityActivityState";
import { storedKeyId } from "@/components/UsagePage/keyActivityData";
import RecommendedReplacement from "@/components/ModelDeprecationAlerts/RecommendedReplacement";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import { entityActivityHref } from "@/utils/entityLinks";
import { copyToClipboard } from "@/utils/dataUtils";
import { hasCapability } from "@/utils/capabilities";
import { uiHref } from "@/utils/uiHref";
import { ApiError } from "@/lib/http/client";
import { useUrlTableState } from "@/components/shared/DataTable";
import {
  Issue,
  IssueStatus,
  issueDate,
  issueImpactText,
  issueHeadline,
  issueResource,
  issuesHref,
  updateIssueStatus,
  useIssue,
} from "./issueApi";
import IssueActivity from "./IssueActivity";
import IssueComparisonTable from "./IssueComparisonTable";
import IssueResources, { ISSUE_RESOURCE_TABLE_STATE } from "./IssueResources";
import IssueSignals from "./IssueSignals";

function resourceHref(issue: Issue): string | undefined {
  const { scope_type: scope, scope_id: id } = issue.cohort;
  if (!id || scope === "provider" || scope === "deployment") return undefined;
  return entityActivityHref(scope, scope === "key" ? storedKeyId(id) : id, {
    from: issue.range_start,
    to: issue.range_end,
  });
}

function IssueStatusActions({ issue }: { issue: Issue }) {
  const { isViewOnly, accessToken } = useAuthorized();
  const { data: user } = useCurrentUser(accessToken);
  const client = useQueryClient();
  const mutation = useMutation({
    mutationFn: (status: IssueStatus) => updateIssueStatus(issue, status),
    onSuccess: (updated) => {
      client.setQueryData(["issues", getProxyBaseUrl(), user?.user_id, user?.user_role, "detail", issue.id], updated);
      return Promise.all([
        client.invalidateQueries({ queryKey: ["issues"] }),
        client.invalidateQueries({ queryKey: ["deprecationAlerts"] }),
      ]);
    },
  });
  const inactiveRetirement = issue.kind === "model_deprecation" && issue.active === false;
  if (isViewOnly || issue.can_update_status !== true) return null;
  if (inactiveRetirement && issue.status !== "open") return null;
  const actions: IssueStatus[] = issue.status === "open" ? ["resolved", "dismissed"] : ["open"];
  const labels = { resolved: "Mark resolved", dismissed: "Dismiss", open: "Reopen" };
  const stale = mutation.error instanceof ApiError && mutation.error.status === 409;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {actions.map((status) => (
          <Button
            key={status}
            variant={status === "dismissed" ? "outline" : "default"}
            size="sm"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate(status)}
          >
            {labels[status]}
          </Button>
        ))}
      </div>
      {mutation.isError && (
        <p role="alert" className="max-w-sm text-xs text-destructive">
          {stale ? "This alert changed. Refresh it before trying again." : "Could not update this alert. Please retry."}
          {stale && (
            <Button
              variant="link"
              size="sm"
              onClick={() => {
                mutation.reset();
                void client.invalidateQueries({ queryKey: ["issues"] });
              }}
            >
              Refresh
            </Button>
          )}
        </p>
      )}
    </div>
  );
}

function RetirementContext({ issue }: { issue: Issue }) {
  const deprecation = issue.deprecation;
  if (!deprecation) return null;
  const copy = () =>
    void copyToClipboard(
      `Please migrate virtual keys using ${deprecation.usage_model}. This model retires on ${deprecation.model.deprecation_date}. Last recorded use: ${deprecation.last_used_date} UTC. Review the affected traffic and confirm migration: ${window.location.href}`,
      "Follow-up copied",
    );
  return (
    <div className="space-y-2">
      <p className="text-sm">Last recorded use: {deprecation.last_used_date} UTC.</p>
      <RecommendedReplacement recommendation={deprecation.model} />
      <div className="flex flex-wrap items-center gap-3 pt-1">
        <Button variant="outline" size="sm" onClick={copy}>
          <Copy className="size-3" />
          Copy follow-up
        </Button>
        <a className="text-sm text-primary hover:underline" href={migrationHref(deprecation.usage_model)}>
          Migration details
        </a>
      </div>
      <p className="text-xs text-muted-foreground">
        Resolve once migration is confirmed across teams. New traffic can reopen this alert.
      </p>
    </div>
  );
}

function IssueBody({ issue }: { issue: Issue }) {
  const [params, setParams] = useQueryStates(entityActivityParsers);
  const resourcesTable = useUrlTableState(ISSUE_RESOURCE_TABLE_STATE);
  const [resourcesOpen, setResourcesOpen] = useState(false);
  const [signalsOpen, setSignalsOpen] = useState(false);
  const from = params.activity_from ?? issue.range_start;
  const to = params.activity_to ?? issue.range_end;
  const href = issue.can_view_activity === true ? resourceHref(issue) : undefined;
  const impact = issueImpactText(issue);
  const updateRange = (from: string | null, to: string | null) => {
    void setParams({ activity_from: from, activity_to: to });
    resourcesTable.onPaginationChange({ ...resourcesTable.pagination, pageIndex: 0 });
  };
  return (
    <div className="min-w-0 space-y-6">
      <header className="flex flex-col items-start gap-4 sm:flex-row sm:justify-between">
        <div className="w-full min-w-0 flex-1 space-y-2 sm:w-auto">
          <h1 className="break-words text-xl font-semibold">{issueHeadline(issue)}</h1>
          <p className="break-all text-sm text-muted-foreground" title={issue.cohort.scope_id ?? "Gateway"}>
            {issueResource(issue)}
          </p>
          <p className="text-xs text-muted-foreground">
            <span className="capitalize">{issue.status}</span> · First detected {issueDate(issue.first_seen)} · Last
            detected {issueDate(issue.last_seen)} UTC
          </p>
        </div>
        <IssueStatusActions issue={issue} />
      </header>
      <section className="space-y-3" aria-label="Alert evidence">
        {issue.kind === "performance" && <p className="text-sm text-muted-foreground">{issue.explanation}</p>}
        <IssueComparisonTable issue={issue} />
        <RetirementContext issue={issue} />
        {impact && <p className="text-xs text-muted-foreground">Recorded resources: {impact}.</p>}
        {href && (
          <a className="text-sm text-primary hover:underline" href={href}>
            View {issue.cohort.scope_type} activity
          </a>
        )}
      </section>
      {issue.can_view_activity === true ? (
        <>
          <section className="space-y-4" aria-label="Alert activity">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-base font-semibold">Activity</h2>
              <div className="flex flex-wrap items-center gap-2">
                {(from !== issue.range_start || to !== issue.range_end) && (
                  <Button size="sm" variant="ghost" onClick={() => updateRange(null, null)}>
                    Reset range
                  </Button>
                )}
                <span className="text-xs text-muted-foreground">UTC</span>
                <AdvancedDatePicker
                  compact
                  label=""
                  className="max-w-xs"
                  presets={activityPresets}
                  showTimeRange={false}
                  value={{ from: new Date(`${from}T00:00:00`), to: new Date(`${to}T23:59:59`) }}
                  onValueChange={(range) => {
                    if (range.from && range.to) updateRange(localActivityDay(range.from), localActivityDay(range.to));
                  }}
                />
              </div>
            </div>
            <IssueActivity issue={issue} from={from} to={to} />
          </section>
          <details className="border-t pt-4" onToggle={(event) => setResourcesOpen(event.currentTarget.open)}>
            <summary className="cursor-pointer text-sm font-medium">Resources in this traffic</summary>
            {resourcesOpen && (
              <div className="pt-4">
                <IssueResources key={`${from}-${to}`} issue={issue} from={from} to={to} />
              </div>
            )}
          </details>
        </>
      ) : (
        <p className="text-sm text-muted-foreground">You don’t have access to this alert’s activity.</p>
      )}
      <details className="border-t pt-4" onToggle={(event) => setSignalsOpen(event.currentTarget.open)}>
        <summary className="cursor-pointer text-sm font-medium">Source signals</summary>
        {signalsOpen && (
          <div className="pt-4">
            <IssueSignals issue={issue} />
          </div>
        )}
      </details>
    </div>
  );
}

export default function IssueDetail({ id }: { id: string }) {
  const { accessToken, userRole } = useAuthorized();
  const query = useIssue(accessToken, id);
  const globalAccess = hasCapability(userRole, "viewGlobalSpend");
  const denied = query.error instanceof ApiError && query.error.status === 403;
  let back = { href: uiHref("usage?view=my-usage"), label: "My usage" };
  if (globalAccess) back = { href: issuesHref(), label: "Alerts" };
  else if (query.data?.can_view_activity === true) {
    const href = resourceHref(query.data);
    if (href) back = { href, label: "Resource activity" };
  }
  let content = (
    <p role="status" className="text-sm text-muted-foreground">
      Loading alert…
    </p>
  );
  if (!accessToken) content = <p>Sign in to view this alert.</p>;
  else if (query.isError)
    content = (
      <div role="alert" className="space-y-2">
        <p>{denied ? "You don’t have access to this alert." : "Could not load this alert."}</p>
        <Button variant="outline" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  else if (query.data) content = <IssueBody key={query.data.id} issue={query.data} />;
  return (
    <div className="min-w-0 space-y-5">
      <a
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        href={back.href}
      >
        <ArrowLeft className="size-4" />
        {back.label}
      </a>
      {content}
    </div>
  );
}
