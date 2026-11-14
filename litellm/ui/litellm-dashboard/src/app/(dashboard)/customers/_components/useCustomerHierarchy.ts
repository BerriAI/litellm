import { useEffect } from "react";
import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { parseAsInteger, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchClient } from "@/lib/http/api";
import i18n from "@/i18n/i18n";
import { isProxyAdminRole } from "@/utils/roles";
import type { components } from "@/lib/http/schema";
import { resolveCustomerHierarchy, type CustomerHierarchyScope } from "./customerHierarchyApi";
import type {
  CreatedCustomerHierarchyEntity,
  CustomerHierarchyEntity,
  Member,
  Organization,
  Team,
} from "./customerHierarchyForms";

const MANAGE_PARSERS = {
  org: parseAsString,
  team: parseAsString,
  member: parseAsString,
  tab: parseAsStringLiteral(["teams", "members", "budget_groups", "settings"] as const),
  hierarchyPage: parseAsInteger.withDefault(1),
};

export interface CustomerHierarchyContext {
  organization: Organization;
  team: Team | null;
  member: Member | null;
  entity: CustomerHierarchyEntity;
}

type ChildList =
  | { kind: "teams"; data: Team[]; meta: components["schemas"]["ListMeta"] }
  | { kind: "members"; data: Member[]; meta: components["schemas"]["ListMeta"] };

const scopeFailure = () => new Error(i18n.t("customerPortal.hierarchy.scopeMismatch"));

function selectedEntity(organization: Organization, team: Team | null, member: Member | null): CustomerHierarchyEntity {
  if (member) return { kind: "member", record: member };
  if (team) return { kind: "team", record: team };
  return { kind: "organization", record: organization };
}

export async function customerHierarchyContext(
  scope: CustomerHierarchyScope,
  accessToken: string,
  signal?: AbortSignal,
): Promise<CustomerHierarchyContext> {
  const selected = await resolveCustomerHierarchy({ scope, accessToken, signal });
  return { ...selected, entity: selectedEntity(selected.organization, selected.team, selected.member) };
}

async function customerHierarchyChildren(
  context: CustomerHierarchyContext,
  kind: ChildList["kind"],
  page: number,
  signal?: AbortSignal,
): Promise<ChildList> {
  if (kind === "members") {
    const query = context.team
      ? { parent_team_id: context.team.customer_team_id, page, page_size: 20 }
      : { parent_org_id: context.organization.customer_org_id, page, page_size: 20 };
    const response = await fetchClient.GET("/management/v1/customer_members", {
      params: { query },
      signal,
    });
    if (
      !response.data ||
      (context.team !== null &&
        response.data.data.some((member) => !member.parent_team_ids.includes(context.team?.customer_team_id ?? "")))
    ) {
      throw scopeFailure();
    }
    return { kind: "members", ...response.data };
  }
  const response = await fetchClient.GET("/management/v1/customer_teams", {
    params: { query: { parent_org_id: context.organization.customer_org_id, page, page_size: 20 } },
    signal,
  });
  if (
    !response.data ||
    response.data.data.some((team) => team.parent_org_id !== context.organization.customer_org_id)
  ) {
    throw scopeFailure();
  }
  return { kind: "teams", ...response.data };
}

function childView(
  selected: CustomerHierarchyContext | undefined,
  requested: "teams" | "members" | "budget_groups" | "settings" | null,
): ChildList["kind"] {
  if (selected?.team) return "members";
  return requested === "members" ? "members" : "teams";
}

function organizationView(
  requested: "teams" | "members" | "budget_groups" | "settings" | null,
  team: string | null,
  children: ChildList["kind"],
) {
  return requested === "budget_groups" && !team ? "budget_groups" : children;
}

export default function useCustomerHierarchy() {
  const { accessToken, userRole, isLoading } = useAuthorized();
  const canManage = isProxyAdminRole(userRole);
  const [params, setParams] = useQueryStates(MANAGE_PARSERS);
  const queryClient = useQueryClient();
  const scope = params.org ? { org: params.org, team: params.team, member: params.member } : null;
  const authenticatedAdmin = Boolean(accessToken && canManage && !isLoading);
  const enabled = authenticatedAdmin && scope !== null;
  const context = useQuery({
    queryKey: ["customer-hierarchy", "manage", params.org, params.team, params.member],
    queryFn:
      enabled && scope && accessToken
        ? ({ signal }) => customerHierarchyContext(scope, accessToken, signal)
        : skipToken,
  });
  const selected = context.data;
  const childTab = childView(selected, params.tab);
  const showSettings = params.member !== null || params.tab === "settings";
  const organizationTab = organizationView(params.tab, params.team, childTab);
  const tab: "teams" | "members" | "budget_groups" | "settings" = showSettings ? "settings" : organizationTab;
  const childContext = selected && !selected.member ? selected : null;
  const showChildren = tab === "teams" || tab === "members";
  const page = Math.max(1, params.hierarchyPage);
  const children = useQuery({
    queryKey: [
      "customer-hierarchy",
      "children",
      childTab,
      selected?.organization.customer_org_id,
      selected?.team?.customer_team_id,
      page,
    ],
    queryFn:
      enabled && childContext && showChildren
        ? ({ signal }) => customerHierarchyChildren(childContext, childTab, page, signal)
        : skipToken,
  });
  useEffect(() => {
    if (!children.isSuccess) return;
    if (page <= Math.max(1, children.data.meta.total_pages)) return;
    void setParams({ hierarchyPage: null });
  }, [children.isSuccess, children.data, page, setParams]);
  const openOrganization = () => {
    const parent = { team: null, member: null, tab: null, hierarchyPage: null };
    void setParams(parent, { history: "push" });
  };
  const openTeam = (id: string, settings = false) => {
    const next = { team: id, member: null, tab: settings ? ("settings" as const) : null, hierarchyPage: null };
    void setParams(next, { history: "push" });
  };
  const openMember = (id: string, teamId: string) => {
    const next = { team: teamId, member: id, tab: null, hierarchyPage: null };
    void setParams(next, { history: "push" });
  };
  const openCreatedScope = (parent: CustomerHierarchyScope, created: CreatedCustomerHierarchyEntity) =>
    void setParams(
      (current) => {
        if (current.org !== parent.org) return {};
        if (current.team !== parent.team) return {};
        if (current.member !== parent.member) return {};
        return {
          org: parent.org,
          team: created.kind === "team" ? created.record.customer_team_id : created.record.parent_team_ids[0] ?? null,
          member: created.kind === "member" ? created.record.customer_member_id : null,
          tab: created.kind === "team" ? ("settings" as const) : null,
          hierarchyPage: null,
        };
      },
      { history: "push" },
    );
  const refresh = () => void queryClient.invalidateQueries({ queryKey: ["customer-hierarchy"] });
  const organizationSaved = async (organization: Organization) => {
    const queryKey = ["customer-hierarchy", "manage", organization.customer_org_id, null, null];
    await queryClient.cancelQueries({ queryKey });
    const saved: CustomerHierarchyContext = {
      organization,
      team: null,
      member: null,
      entity: { kind: "organization", record: organization },
    };
    queryClient.setQueryData(queryKey, saved);
    refresh();
  };
  return {
    scope,
    context,
    children,
    selected,
    tab,
    childTab,
    page,
    canManage,
    isLoading,
    openOrganization,
    openTeam,
    openMember,
    openCreatedScope,
    refresh,
    organizationSaved,
    setPage: (hierarchyPage: number) => void setParams({ hierarchyPage }),
  };
}
