import { useTranslation } from "react-i18next";
import { createParser, createSerializer, parseAsString, useQueryStates, type inferParserType } from "nuqs";
import { uiHref } from "@/utils/uiHref";
import type { CustomerUsageScope } from "./customerUsageApi";
import { customerUsageDateParsers } from "./customerUsageRange";
import { CustomerDetailNavigation } from "./CustomerDetailLayout";

type CustomerHierarchyView = "usage" | "teams" | "members" | "budget_groups" | "settings";
const LABELS = {
  usage: "customerPortal.hierarchy.usage",
  teams: "customerPortal.hierarchy.teams",
  members: "customerPortal.hierarchy.members",
  budget_groups: "customerPortal.budgetGroups.title",
  settings: "customerPortal.hierarchy.settings",
} as const;

export const customerAppParsers = {
  app_id: createParser({
    parse: (value) => (value.length > 0 && !value.includes(",") ? value : null),
    serialize: (value: string) => value,
  }),
};
const hierarchyFilters = { ...customerUsageDateParsers, ...customerAppParsers };
const hierarchyParsers = {
  org: parseAsString,
  team: parseAsString,
  member: parseAsString,
  ...hierarchyFilters,
};
const hierarchyHref = createSerializer(hierarchyParsers);
const listHref = createSerializer(customerUsageDateParsers);

export function customerHierarchyHref(
  page: "usage" | "manage",
  scope: CustomerUsageScope,
  dates: Partial<inferParserType<typeof hierarchyFilters>> = {},
): string {
  return hierarchyHref(uiHref(`customers/${page}`), { ...scope, ...dates });
}

export function useCustomerHierarchyLinks() {
  const [dates] = useQueryStates(hierarchyFilters);
  return {
    customerHierarchyHref: (page: "usage" | "manage", scope: CustomerUsageScope) =>
      customerHierarchyHref(page, scope, dates),
    customersHref: listHref(uiHref("customers"), dates),
  };
}

export default function CustomerHierarchyNavigation({
  scope,
  active,
}: {
  scope: CustomerUsageScope;
  active: CustomerHierarchyView;
}) {
  const { t } = useTranslation();
  const { customerHierarchyHref } = useCustomerHierarchyLinks();
  let pages: readonly CustomerHierarchyView[] = ["usage", "teams", "members", "budget_groups", "settings"];
  if (scope.team) pages = ["usage", "members", "settings"];
  if (scope.member) pages = ["usage", "settings"];
  const items = pages.map((page) => {
    const organizationMembers = page === "members" && !scope.team;
    const explicitTab = page === "settings" || page === "budget_groups" || organizationMembers;
    return {
      href:
        page === "usage"
          ? customerHierarchyHref("usage", scope)
          : `${customerHierarchyHref("manage", scope)}${explicitTab ? `&tab=${page}` : ""}`,
      label: t(LABELS[page]),
      active: active === page,
    };
  });
  return <CustomerDetailNavigation label={t("customerPortal.hierarchy.customerNavigation")} items={items} />;
}
