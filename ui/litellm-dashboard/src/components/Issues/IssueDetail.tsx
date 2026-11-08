import { useState } from "react";
import { useQueryStates } from "nuqs";
import { ArrowLeft, Copy } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button, buttonVariants } from "@/components/ui/button";
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
  issueDate,
  issueComparisonText,
  issueImpactText,
  issueHeadline,
  issueResource,
  issuesHref,
  useIssue,
} from "./issueApi";
import IssueActivity from "./IssueActivity";
import IssueComparisonTable from "./IssueComparisonTable";
import IssueResources, { ISSUE_RESOURCE_TABLE_STATE } from "./IssueResources";
import IssueSignals from "./IssueSignals";
import IssueStatusActions from "./IssueStatusActions";
import IssueHistoryChart from "./IssueHistoryChart";

function resourceHref(issue: Issue): string | undefined {
  const { scope_type: scope, scope_id: id } = issue.cohort;
  if (!id || scope === "provider" || scope === "deployment") return undefined;
  return entityActivityHref(scope, scope === "key" ? storedKeyId(id) : id, {
    from: issue.range_start,
    to: issue.range_end,
  });
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
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [activityOpen, setActivityOpen] = useState(false);
  const from = params.activity_from ?? issue.range_start;
  const to = params.activity_to ?? issue.range_end;
  const href = issue.can_view_activity === true ? resourceHref(issue) : undefined;
  const impact = issueImpactText(issue);
  const updateRange = (from: string | null, to: string | null) => {
    void setParams({ activity_from: from, activity_to: to });
    resourcesTable.onPaginationChange({ ...resourcesTable.pagination, pageIndex: 0 });
  };
  return (
    <div className="min-w-0 space-y-7">
      <header className="max-w-4xl space-y-3">
        <h1 className="break-words text-2xl font-semibold leading-snug">{issueHeadline(issue)}</h1>
        <p className="text-base leading-relaxed text-muted-foreground">{issueComparisonText(issue)}</p>
      </header>
      <div className="grid min-w-0 gap-8 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0 space-y-6">
          <IssueHistoryChart issue={issue} />
          {issue.can_view_activity === true ? (
            <>
              {!href && (
                <details
                  open={activityOpen}
                  className="border-t pt-4"
                  onToggle={(event) => setActivityOpen(event.currentTarget.open)}
                >
                  <summary className="cursor-pointer text-sm font-medium">View activity for this alert</summary>
                  {activityOpen && (
                    <div className="pt-4">
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
                                if (range.from && range.to)
                                  updateRange(localActivityDay(range.from), localActivityDay(range.to));
                              }}
                            />
                          </div>
                        </div>
                        <IssueActivity issue={issue} from={from} to={to} />
                      </section>
                    </div>
                  )}
                </details>
              )}
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
          <details className="border-t pt-4" onToggle={(event) => setDetailsOpen(event.currentTarget.open)}>
            <summary className="cursor-pointer text-sm font-medium">Alert details</summary>
            {detailsOpen && (
              <div className="space-y-4 pt-4">
                <p className="text-xs text-muted-foreground">
                  First detected {issueDate(issue.first_seen)} · Last detected {issueDate(issue.last_seen)} UTC
                </p>
                <p className="text-sm text-muted-foreground">{issue.explanation}</p>
                <IssueComparisonTable issue={issue} />
                <IssueSignals issue={issue} />
              </div>
            )}
          </details>
        </div>
        <aside className="space-y-6 border-t pt-6 lg:border-t-0 lg:border-l lg:pt-0 lg:pl-6" aria-label="Alert context">
          <section className="space-y-3">
            <h2 className="text-sm font-semibold">{issue.deprecation ? "Plan the migration" : "Investigate"}</h2>
            {issue.deprecation ? (
              <RetirementContext issue={issue} />
            ) : (
              <p className="text-sm leading-6 text-muted-foreground">
                Review the recorded traffic to see what changed across models and requests.
              </p>
            )}
            {href && (
              <a className={buttonVariants({ size: "sm", variant: "outline" })} href={href}>
                View {issue.cohort.scope_type} activity
              </a>
            )}
            {!href && issue.can_view_activity === true && (
              <Button variant="outline" size="sm" onClick={() => setActivityOpen(true)}>
                View activity for this alert
              </Button>
            )}
          </section>
          <section className="space-y-3 border-t pt-5">
            <h2 className="text-sm font-semibold">Scope</h2>
            <p className="break-words text-sm">{issueResource(issue)}</p>
            {impact && <p className="text-sm text-muted-foreground">{impact}</p>}
            <dl className="space-y-3 text-xs text-muted-foreground">
              <div>
                <dt>Observed</dt>
                <dd className="mt-1 text-foreground">{issueDate(issue.observed_on)} UTC</dd>
              </div>
              <div>
                <dt>First detected</dt>
                <dd className="mt-1 text-foreground">{issueDate(issue.first_seen)} UTC</dd>
              </div>
            </dl>
          </section>
          <section className="space-y-3 border-t pt-5">
            <p className="text-sm capitalize">{issue.status}</p>
            <IssueStatusActions issue={issue} />
          </section>
        </aside>
      </div>
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
