import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { apiClient } from "@/components/networking";
import { customerFromHeaderPairs, customerHeaderPairs } from "@/components/llm_calls/request_headers";
import type { KeyValuePair } from "@/components/key_value_input";
import type { components } from "@/lib/http/schema";
import { resolveCustomerHierarchy } from "../../../customers/_components/customerHierarchyApi";
import { CUSTOMER_CONTEXT_PARSERS, customerRequestFailure, safeCustomerValue } from "./customerContext";
import type { PlaygroundSettings } from "./playgroundSettings";

type Organization = components["schemas"]["Organization"];
type Team = components["schemas"]["Team"];
type Member = components["schemas"]["CustomerMember"];
type Page<T> = Readonly<{ data: readonly T[]; meta: Readonly<{ total_pages: number }> }>;
type Collection = "customer_organizations" | "customer_teams" | "customer_members";

async function allPages<T>(
  resource: Collection,
  token: string,
  signal: AbortSignal,
  options: Readonly<{ filter?: Readonly<Record<string, string>>; page?: number }> = {},
): Promise<readonly T[]> {
  const { filter = {}, page = 1 } = options;
  const response = await apiClient.get<Page<T>>(`/management/v1/${resource}`, {
    accessToken: token,
    signal,
    query: { ...filter, page, page_size: 100 },
  });
  return page >= response.meta.total_pages
    ? response.data
    : [...response.data, ...(await allPages<T>(resource, token, signal, { filter, page: page + 1 }))];
}

async function membersInCustomerOrg(orgId: string, token: string, signal: AbortSignal): Promise<readonly Member[]> {
  const teams = await allPages<Team>("customer_teams", token, signal, { filter: { parent_org_id: orgId } });
  const members = await Promise.all(
    teams.map((team) =>
      allPages<Member>("customer_members", token, signal, { filter: { parent_team_id: team.customer_team_id } }),
    ),
  );
  return members.flat();
}

export function usePlaygroundCustomer(
  accessToken: string | null,
  headerPairs: readonly KeyValuePair[],
  update: (next: Partial<PlaygroundSettings> | ((current: PlaygroundSettings) => PlaygroundSettings)) => void,
) {
  const { t } = useTranslation();
  const [query, setQuery] = useQueryStates(CUSTOMER_CONTEXT_PARSERS);
  const [link] = useState(() => query);
  const [pendingLink, setPendingLink] = useState(Boolean(link.customerOrg || link.customerMember));
  const [refreshUntil, setRefreshUntil] = useState(0);
  const [lastFailure, setLastFailure] = useState<Readonly<{
    scope: string;
    failure: ReturnType<typeof customerRequestFailure>;
  }> | null>(null);
  const headers = customerFromHeaderPairs(headerPairs);
  const organizations = useQuery({
    queryKey: ["playgroundCustomers", "organizations", accessToken],
    queryFn: ({ signal }) => allPages<Organization>("customer_organizations", accessToken!, signal),
    enabled: Boolean(accessToken),
  });
  const matches = organizations.data?.filter((org) => org.alias === headers.alias) ?? [];
  const organization = matches.length === 1 ? matches[0] : undefined;
  const organizationId = organization?.customer_org_id;
  const members = useQuery({
    queryKey: ["playgroundCustomers", "members", organizationId, accessToken],
    queryFn: ({ signal }) => membersInCustomerOrg(organization!.customer_org_id, accessToken!, signal),
    enabled: Boolean(organization && accessToken),
  });
  const initialOptions = {
    queryKey: ["playgroundCustomers", "link", link.customerOrg, link.customerMember, accessToken],
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const result = await resolveCustomerHierarchy({
        scope: { org: link.customerOrg!, team: null, member: link.customerMember },
        accessToken: accessToken!,
        signal,
      });
      if (!result.organization.alias || !safeCustomerValue(result.organization.alias))
        throw new Error(t("playground.chat.customer.aliasRequired"));
      return result;
    },
    enabled: Boolean(pendingLink && link.customerOrg && accessToken),
    retry: false,
  };
  const initial = useQuery(initialOptions);
  useEffect(() => {
    if (!pendingLink || initial.isError) void setQuery({ customerOrg: null, customerMember: null });
  }, [initial.isError, pendingLink, setQuery]);
  if (pendingLink && initial.data) {
    const { organization: org, member } = initial.data;
    update((current) => ({
      ...current,
      headerPairs: customerHeaderPairs(current.headerPairs, org.alias ?? null, member?.customer_member_id ?? null),
    }));
    setPendingLink(false);
  }
  const scopeKey = JSON.stringify([organizationId, headers.memberId]);
  const budgetOptions = {
    queryKey: ["playgroundCustomers", "budget", organizationId, headers.memberId, accessToken],
    queryFn: ({ signal }: { signal: AbortSignal }) =>
      resolveCustomerHierarchy({
        scope: { org: organization!.customer_org_id, team: null, member: headers.memberId },
        accessToken: accessToken!,
        signal,
      }),
    enabled: Boolean(organization && accessToken && !pendingLink),
    retry: false,
    refetchInterval: (): number | false => (Date.now() < refreshUntil ? 2000 : false),
  };
  const budget = useQuery(budgetOptions);
  const setContext = (org: Organization | null, memberId: string | null) => {
    setPendingLink(false);
    update((current) => ({
      ...current,
      headerPairs: customerHeaderPairs(current.headerPairs, org?.alias ?? null, memberId),
    }));
  };
  const onRequestSettled = (error: unknown) => {
    if (!headers.alias || !organization) return;
    setLastFailure({ scope: scopeKey, failure: customerRequestFailure(error) });
    setRefreshUntil(Date.now() + 60_000);
    void budget.refetch();
  };
  return {
    organizations,
    organization,
    members,
    memberId: headers.memberId,
    alias: headers.alias,
    setContext,
    budget,
    pendingLink,
    onRequestSettled,
    failure: lastFailure?.scope === scopeKey ? lastFailure.failure : null,
    linkError:
      pendingLink && (initial.isError || !link.customerOrg)
        ? initial.error ?? new Error(t("playground.chat.customer.orgRequired"))
        : null,
  };
}
