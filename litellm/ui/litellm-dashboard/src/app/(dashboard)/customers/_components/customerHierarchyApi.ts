import { apiClient } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import i18n from "@/i18n/i18n";
import { $api, fetchClient } from "@/lib/http/api";
import type { CustomerHierarchyPatch } from "./customerHierarchyForms";

type Organization = components["schemas"]["Organization"];
type Team = components["schemas"]["Team"];
type Member = components["schemas"]["CustomerMember"];
type Collection = "customer_organizations" | "customer_teams" | "customer_members";

export async function saveCustomerHierarchy(id: string, patch: CustomerHierarchyPatch) {
  switch (patch.kind) {
    case "organization": {
      const params = { path: { customer_org_id: id } };
      const { data } = await fetchClient.PATCH("/management/v1/customer_organizations/{customer_org_id}", {
        params,
        body: patch.body,
      });
      return {
        data,
        queryKey: $api.queryOptions("get", "/management/v1/customer_organizations/{customer_org_id}", { params })
          .queryKey,
      };
    }
    case "team": {
      const params = { path: { customer_team_id: id } };
      const { data } = await fetchClient.PATCH("/management/v1/customer_teams/{customer_team_id}", {
        params,
        body: patch.body,
      });
      return {
        data,
        queryKey: $api.queryOptions("get", "/management/v1/customer_teams/{customer_team_id}", { params }).queryKey,
      };
    }
    case "member": {
      const params = { path: { customer_member_id: id } };
      const { data } = await fetchClient.PATCH("/management/v1/customer_members/{customer_member_id}", {
        params,
        body: patch.body,
      });
      return {
        data,
        queryKey: $api.queryOptions("get", "/management/v1/customer_members/{customer_member_id}", { params }).queryKey,
      };
    }
  }
}

export interface CustomerHierarchyScope {
  org: string;
  team: string | null;
  member: string | null;
}

export interface ResolvedCustomerHierarchy {
  organization: Organization;
  team: Team | null;
  member: Member | null;
}

interface HierarchyRequest {
  scope: CustomerHierarchyScope;
  accessToken: string;
  signal?: AbortSignal;
}

const assertScope = (valid: boolean) => {
  if (!valid) throw new Error(i18n.t("customerPortal.analytics.scopeMismatch"));
};

const info = <T>(resource: Collection, id: string, request: HierarchyRequest) =>
  apiClient.get<T>(`/management/v1/${resource}/${encodeURIComponent(id)}`, {
    accessToken: request.accessToken,
    signal: request.signal,
  });

export const resolveCustomerHierarchy = async (request: HierarchyRequest): Promise<ResolvedCustomerHierarchy> => {
  const { scope } = request;
  const [organization, requestedTeam, member] = await Promise.all([
    info<Organization>("customer_organizations", scope.org, request),
    scope.team ? info<Team>("customer_teams", scope.team, request) : null,
    scope.member ? info<Member>("customer_members", scope.member, request) : null,
  ]);
  const team =
    requestedTeam ?? (member ? await info<Team>("customer_teams", member.parent_team_ids[0], request) : null);
  assertScope(organization.customer_org_id === scope.org);
  assertScope(!team || team.parent_org_id === scope.org);
  assertScope(!scope.team || team?.customer_team_id === scope.team);
  assertScope(!member || member.customer_member_id === scope.member);
  assertScope(!member || (team !== null && member.parent_team_ids.includes(team.customer_team_id)));
  return { organization, team, member };
};
