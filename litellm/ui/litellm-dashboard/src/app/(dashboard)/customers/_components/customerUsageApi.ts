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
  budgetResetAt: string | null;
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
}

interface CustomerUsageRequest {
  scope: CustomerUsageScope;
  accessToken: string;
  startDate: string;
  endDate: string;
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
        if (members.some((member) => member.parent_team_id !== team.customer_team_id))
          throw new Error(i18n.t("customerPortal.analytics.scopeMismatch"));
        return members;
      });
  return {
    teams: [...new Map(teams.map((team) => [team.customer_team_id, team])).values()],
    members: [...new Map(memberLists.flat().map((member) => [member.customer_member_id, member])).values()],
  };
};

const activityPages = async (resource: Collection, ids: readonly string[], request: CustomerUsageRequest) => {
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

interface ActivityJob {
  dimension: "scope" | "teams" | "members";
  resource: Collection;
  ids: readonly string[];
}

const chunkJobs = (resource: Collection, dimension: ActivityJob["dimension"], ids: readonly string[]): ActivityJob[] =>
  Array.from({ length: Math.ceil(ids.length / PAGE_SIZE) }, (_, index) => ({
    resource,
    dimension,
    ids: ids.slice(index * PAGE_SIZE, (index + 1) * PAGE_SIZE),
  }));

const entity = (id: string, record: Organization | Team | Member): CustomerUsageEntity => ({
  id,
  label: record.alias || id,
  spend: record.spend,
  budgetLimit: ("effective_budget" in record ? record.effective_budget : record.budget)?.max_budget ?? null,
  budgetResetAt: record.budget_reset_at ?? null,
});

const selectedResource = (scope: ResolvedScope): Collection => {
  if (scope.member) return "customer_members";
  if (scope.team) return "customer_teams";
  return "customer_organizations";
};

export const customerHierarchyUsageCall = async (request: CustomerUsageRequest): Promise<CustomerHierarchyUsage> => {
  assertIdentifiers([
    request.scope.org,
    ...[request.scope.team, request.scope.member].filter((id): id is string => id !== null),
  ]);
  const selected = await resolveCustomerHierarchy(request);
  const children = await descendants(selected, request);
  const resource = selectedResource(selected);
  const selectedId = selected.member?.customer_member_id ?? selected.team?.customer_team_id ?? request.scope.org;
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
  ];
  const activity = await batches(jobs, async (job) => ({
    dimension: job.dimension,
    pages: await activityPages(job.resource, job.ids, request),
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
  const teamLabels = new Map(
    children.teams.map((team) => [team.customer_team_id, team.alias || team.customer_team_id]),
  );
  const memberLabels = new Map(
    children.members.map((member) => [member.customer_member_id, member.alias || member.customer_member_id]),
  );
  return {
    ...primary,
    organization: entity(selected.organization.customer_org_id, selected.organization),
    team: selected.team ? entity(selected.team.customer_team_id, selected.team) : null,
    member: selected.member ? entity(selected.member.customer_member_id, selected.member) : null,
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
    memberBudgets: children.members.map((member) => entity(member.customer_member_id, member)),
  };
};
