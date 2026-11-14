import { apiClient } from "@/components/networking";
import { type DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import type { components } from "@/lib/http/schema";
import i18n from "@/i18n/i18n";
import {
  resolveCustomerHierarchy,
  type CustomerHierarchyScope as CustomerUsageScope,
  type ResolvedCustomerHierarchy as ResolvedScope,
} from "./customerHierarchyApi";
export type { CustomerHierarchyScope as CustomerUsageScope } from "./customerHierarchyApi";
import {
  mergeCustomerActivityPages,
  customerEntityRankings,
  customerKeyRankings,
  customerEntityDailyData,
  type CustomerUsageRow,
} from "./customerUsageData";

type Organization = components["schemas"]["Organization"];
type Team = components["schemas"]["Team"];
type Member = components["schemas"]["CustomerMember"];
type Collection = "customer_organizations" | "customer_teams" | "customer_members";
type UsageRow = Omit<CustomerUsageRow, "share">;

export interface CustomerUsageEntity {
  id: string;
  label: string;
  spend: number;
  budgetLimit: number | null;
  budgetDuration: string | null;
  budgetLimits: components["schemas"]["BudgetLimit"][];
  budgetResetAt: string | null;
  budgetUsage?: components["schemas"]["CustomerBudgetUsage"][] | null;
}

export interface CustomerHierarchyUsage extends ReturnType<typeof mergeCustomerActivityPages> {
  organization: CustomerUsageEntity;
  team: CustomerUsageEntity | null;
  member: CustomerUsageEntity | null;
  teams: UsageRow[];
  members: UsageRow[];
  api_keys: UsageRow[];
  teamResults: ReturnType<typeof mergeCustomerActivityPages>["results"];
  memberResults: ReturnType<typeof mergeCustomerActivityPages>["results"];
  memberBudgets: CustomerUsageEntity[];
  monthlyOrganizationSpend: number;
  monthlyMembers: UsageRow[];
  yearlyMembers: UsageRow[];
}

interface CustomerUsageRequest {
  scope: CustomerUsageScope;
  accessToken: string;
  startDate: string;
  endDate: string;
  monthStartDate: string;
  monthEndDate: string;
  yearStartDate: string;
  yearEndDate: string;
  signal?: AbortSignal;
}

const PREFIX = "/management/v1";
const PAGE_SIZE = 100;
const MAX_PAGES = 1_000;
const MAX_CONCURRENT = 4;
const ID_FILTER: Record<Collection, string> = {
  customer_organizations: "customer_org_ids",
  customer_teams: "customer_team_ids",
  customer_members: "customer_member_ids",
};

const assertIdentifiers = (ids: readonly string[]) => {
  if (ids.some((id) => !id || id.includes(","))) throw new Error(i18n.t("customerPortal.analytics.invalidIdentifier"));
};

const batches = async <T, R>(items: readonly T[], task: (item: T) => Promise<R>): Promise<R[]> => {
  const results: R[] = [];
  for (let start = 0; start < items.length; start += MAX_CONCURRENT) {
    results.push(...(await Promise.all(items.slice(start, start + MAX_CONCURRENT).map(task))));
  }
  return results;
};

const listAll = async <T>(
  resource: Collection,
  parent: Record<string, string>,
  request: CustomerUsageRequest,
): Promise<T[]> => {
  const rows: T[] = [];
  for (let page = 1; page <= MAX_PAGES; page++) {
    const result = await apiClient.get<{ data: T[]; meta: components["schemas"]["ListMeta"] }>(
      `${PREFIX}/${resource}`,
      {
        accessToken: request.accessToken,
        signal: request.signal,
        query: { ...parent, page, page_size: PAGE_SIZE },
      },
    );
    rows.push(...result.data);
    if (page >= result.meta.total_pages) return rows;
  }
  throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
};

const descendants = async (scope: ResolvedScope, request: CustomerUsageRequest) => {
  const teams = scope.team
    ? [scope.team]
    : await listAll<Team>("customer_teams", { parent_org_id: request.scope.org }, request);
  if (teams.some((team) => team.parent_org_id !== request.scope.org))
    throw new Error(i18n.t("customerPortal.analytics.scopeMismatch"));
  const memberLists = scope.member
    ? [[scope.member]]
    : await batches(teams, async (team) => {
        const members = await listAll<Member>("customer_members", { parent_team_id: team.customer_team_id }, request);
        if (members.some((member) => !member.parent_team_ids.includes(team.customer_team_id)))
          throw new Error(i18n.t("customerPortal.analytics.scopeMismatch"));
        return members;
      });
  return {
    teams: [...new Map(teams.map((team) => [team.customer_team_id, team])).values()],
    members: [...new Map(memberLists.flat().map((member) => [member.customer_member_id, member])).values()],
  };
};

const activityPages = async (
  resource: Collection,
  ids: readonly string[],
  request: Pick<CustomerUsageRequest, "accessToken" | "startDate" | "endDate" | "signal">,
) => {
  assertIdentifiers(ids);
  const pages: DailyActivityAggregatedResponse[] = [];
  for (let page = 1; page <= MAX_PAGES; page++) {
    const result = await apiClient.get<DailyActivityAggregatedResponse>(`${PREFIX}/${resource}/daily/activity`, {
      accessToken: request.accessToken,
      signal: request.signal,
      query: {
        start_date: request.startDate,
        end_date: request.endDate,
        [ID_FILTER[resource]]: ids.join(","),
        page,
        page_size: PAGE_SIZE,
      },
    });
    if (!result.metadata) throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
    pages.push(result);
    if (!result.metadata.has_more && page >= result.metadata.total_pages) return pages;
  }
  throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
};

const organizationActivityTotals = async (
  organizationId: string,
  request: Pick<CustomerUsageRequest, "accessToken" | "startDate" | "endDate" | "signal">,
) => {
  const result = await apiClient.get<DailyActivityAggregatedResponse>(
    `${PREFIX}/customer_organizations/daily/activity/aggregated`,
    {
      accessToken: request.accessToken,
      signal: request.signal,
      query: {
        start_date: request.startDate,
        end_date: request.endDate,
        customer_org_id: organizationId,
      },
    },
  );
  if (!result.metadata) throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
  return { ...result, metadata: result.metadata };
};

export const customerChildSpendCall = async ({
  organizationId,
  kind,
  entities,
  ...request
}: Pick<
  CustomerUsageRequest,
  "accessToken" | "monthStartDate" | "monthEndDate" | "yearStartDate" | "yearEndDate" | "signal"
> & {
  organizationId: string;
  kind: "teams" | "members";
  entities: readonly CustomerUsageEntity[];
}) => {
  const ids = entities.map((item) => item.id);
  if (ids.length === 0) return { monthly: [], yearly: [], organizationSpend: 0 };
  if (!organizationId) throw new Error(i18n.t("customerPortal.analytics.invalidIdentifier"));
  assertIdentifiers(ids);
  const resource = kind === "teams" ? "customer_teams" : "customer_members";
  const monthRequest = { ...request, startDate: request.monthStartDate, endDate: request.monthEndDate };
  const yearRequest = { ...request, startDate: request.yearStartDate, endDate: request.yearEndDate };
  const [monthlyPages, yearlyPages, organizationActivity] = await Promise.all([
    activityPages(resource, ids, monthRequest),
    activityPages(resource, ids, yearRequest),
    organizationActivityTotals(organizationId, monthRequest),
  ]);
  const monthly = mergeCustomerActivityPages(monthlyPages);
  const yearly = mergeCustomerActivityPages(yearlyPages);
  const labels = new Map(entities.map((item) => [item.id, item.label]));
  const options = { unassignedLabel: "" };
  return {
    monthly: customerEntityRankings(monthly.results, labels, monthly.metadata, options),
    yearly: customerEntityRankings(yearly.results, labels, yearly.metadata, options),
    organizationSpend: organizationActivity.metadata.total_spend,
  };
};

interface ActivityJob {
  dimension: "scope" | "teams" | "members" | "monthlyOrganization" | "monthlyMembers" | "yearlyMembers";
  resource: Collection;
  ids: readonly string[];
}

const chunkJobs = (resource: Collection, dimension: ActivityJob["dimension"], ids: readonly string[]): ActivityJob[] =>
  Array.from({ length: Math.ceil(ids.length / PAGE_SIZE) }, (_, index) => ({
    resource,
    dimension,
    ids: ids.slice(index * PAGE_SIZE, (index + 1) * PAGE_SIZE),
  }));

export const customerUsageEntity = (id: string, record: Organization | Team | Member): CustomerUsageEntity => {
  const budget = "effective_budget" in record ? record.effective_budget : record.budget;
  return {
    id,
    label: record.alias || id,
    spend: record.spend,
    budgetLimit: budget?.max_budget ?? null,
    budgetDuration: budget?.budget_duration ?? null,
    budgetLimits: budget?.budget_limits ?? [],
    budgetResetAt: record.budget_reset_at ?? null,
    budgetUsage: record.budget_usage,
  };
};

const selectedResource = (scope: ResolvedScope): Collection => {
  if (scope.member) return "customer_members";
  if (scope.team) return "customer_teams";
  return "customer_organizations";
};

const monthlyJobs = (
  isCurrentMonth: boolean,
  resource: Collection,
  organizationId: string,
  memberIds: readonly string[],
): ActivityJob[] => [
  ...(!isCurrentMonth || resource !== "customer_organizations"
    ? chunkJobs("customer_organizations", "monthlyOrganization", [organizationId])
    : []),
  ...(!isCurrentMonth ? chunkJobs("customer_members", "monthlyMembers", memberIds) : []),
];

const isSelectedRange = (request: CustomerUsageRequest, startDate: string, endDate: string): boolean =>
  request.startDate === startDate && request.endDate === endDate;

export const customerHierarchyUsageCall = async (request: CustomerUsageRequest): Promise<CustomerHierarchyUsage> => {
  assertIdentifiers([
    request.scope.org,
    ...[request.scope.team, request.scope.member].filter((id): id is string => id !== null),
  ]);
  const selected = await resolveCustomerHierarchy(request);
  const children = await descendants(selected, request);
  const resource = selectedResource(selected);
  const selectedId = selected.member?.customer_member_id ?? selected.team?.customer_team_id ?? request.scope.org;
  const isCurrentMonth = isSelectedRange(request, request.monthStartDate, request.monthEndDate);
  const isCurrentYear = isSelectedRange(request, request.yearStartDate, request.yearEndDate);
  const jobs: ActivityJob[] = [
    { dimension: "scope", resource, ids: [selectedId] },
    ...(!selected.team
      ? chunkJobs(
          "customer_teams",
          "teams",
          children.teams.map((team) => team.customer_team_id),
        )
      : []),
    ...(!selected.member
      ? chunkJobs(
          "customer_members",
          "members",
          children.members.map((member) => member.customer_member_id),
        )
      : []),
    ...monthlyJobs(
      isCurrentMonth,
      resource,
      request.scope.org,
      children.members.map((member) => member.customer_member_id),
    ),
    ...(!isCurrentYear
      ? chunkJobs(
          "customer_members",
          "yearlyMembers",
          children.members.map((member) => member.customer_member_id),
        )
      : []),
  ];
  const monthRequest = { ...request, startDate: request.monthStartDate, endDate: request.monthEndDate };
  const yearRequest = { ...request, startDate: request.yearStartDate, endDate: request.yearEndDate };
  const requests: Record<ActivityJob["dimension"], CustomerUsageRequest> = {
    scope: request,
    teams: request,
    members: request,
    monthlyOrganization: monthRequest,
    monthlyMembers: monthRequest,
    yearlyMembers: yearRequest,
  };
  const activity = await batches(jobs, async (job) => ({
    dimension: job.dimension,
    pages:
      job.dimension === "monthlyOrganization"
        ? [await organizationActivityTotals(request.scope.org, monthRequest)]
        : await activityPages(job.resource, job.ids, requests[job.dimension]),
  }));
  const primary = mergeCustomerActivityPages(
    activity.filter((result) => result.dimension === "scope").flatMap((result) => result.pages),
  );
  const teamDays = selected.team
    ? primary.results
    : mergeCustomerActivityPages(
        activity.filter((result) => result.dimension === "teams").flatMap((result) => result.pages),
      ).results;
  const memberDays = selected.member
    ? primary.results
    : mergeCustomerActivityPages(
        activity.filter((result) => result.dimension === "members").flatMap((result) => result.pages),
      ).results;
  const monthlyOrganization =
    isCurrentMonth && resource === "customer_organizations"
      ? primary
      : mergeCustomerActivityPages(
          activity.filter((result) => result.dimension === "monthlyOrganization").flatMap((result) => result.pages),
        );
  const monthlyMemberDays = isCurrentMonth
    ? memberDays
    : mergeCustomerActivityPages(
        activity.filter((result) => result.dimension === "monthlyMembers").flatMap((result) => result.pages),
      ).results;
  const yearlyMemberActivity = isCurrentYear
    ? { results: memberDays, metadata: primary.metadata }
    : mergeCustomerActivityPages(
        activity.filter((result) => result.dimension === "yearlyMembers").flatMap((result) => result.pages),
      );
  const teamLabels = new Map(
    children.teams.map((team) => [team.customer_team_id, team.alias || team.customer_team_id]),
  );
  const memberLabels = new Map(
    children.members.map((member) => [member.customer_member_id, member.alias || member.customer_member_id]),
  );
  return {
    ...primary,
    organization: customerUsageEntity(selected.organization.customer_org_id, selected.organization),
    team: selected.team ? customerUsageEntity(selected.team.customer_team_id, selected.team) : null,
    member: selected.member ? customerUsageEntity(selected.member.customer_member_id, selected.member) : null,
    teams: customerEntityRankings(teamDays, teamLabels, primary.metadata, {
      unassignedLabel: i18n.t("customerPortal.analytics.unassignedTeam"),
      onlyEntityId: selected.member ? selected.team?.customer_team_id : undefined,
    }),
    members: customerEntityRankings(memberDays, memberLabels, primary.metadata, {
      unassignedLabel: i18n.t("customerPortal.analytics.unassignedMember"),
    }),
    api_keys: customerKeyRankings(primary.results, i18n.t("customerPortal.analytics.unnamedKey")),
    teamResults: customerEntityDailyData(
      primary.results,
      teamDays,
      teamLabels,
      selected.member ? selected.team?.customer_team_id : undefined,
    ),
    memberResults: customerEntityDailyData(primary.results, memberDays, memberLabels),
    memberBudgets: children.members.map((member) => customerUsageEntity(member.customer_member_id, member)),
    monthlyOrganizationSpend: monthlyOrganization.metadata.total_spend,
    monthlyMembers: customerEntityRankings(monthlyMemberDays, memberLabels, monthlyOrganization.metadata, {
      unassignedLabel: i18n.t("customerPortal.analytics.unassignedMember"),
    }),
    yearlyMembers: customerEntityRankings(yearlyMemberActivity.results, memberLabels, yearlyMemberActivity.metadata, {
      unassignedLabel: i18n.t("customerPortal.analytics.unassignedMember"),
    }),
  };
};
