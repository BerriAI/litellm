import { useTranslation } from "react-i18next";
import { createSerializer, parseAsString, useQueryStates, type inferParserType } from "nuqs";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import type { CustomerUsageScope } from "./customerUsageApi";
import { customerUsageDateParsers } from "./customerUsageRange";

type CustomerHierarchyView = "usage" | "teams" | "members" | "budget_groups" | "settings";
const LABELS = {
  usage: "customerPortal.hierarchy.usage",
  teams: "customerPortal.hierarchy.teams",
  members: "customerPortal.hierarchy.members",
  budget_groups: "customerPortal.budgetGroups.title",
  settings: "customerPortal.hierarchy.settings",
} as const;

const hierarchyParsers = {
  org: parseAsString,
  team: parseAsString,
  member: parseAsString,
  ...customerUsageDateParsers,
};
const hierarchyHref = createSerializer(hierarchyParsers);
const listHref = createSerializer(customerUsageDateParsers);

export function customerHierarchyHref(
  page: "usage" | "manage",
  scope: CustomerUsageScope,
  dates: Partial<inferParserType<typeof customerUsageDateParsers>> = {},
): string {
  return hierarchyHref(uiHref(`customers/${page}`), { ...scope, ...dates });
}

export function useCustomerHierarchyLinks() {
  const [dates] = useQueryStates(customerUsageDateParsers);
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
  return (
    <nav aria-label={t("customerPortal.hierarchy.customerNavigation")} className="flex gap-6 border-b">
      {pages.map((page) => {
        const organizationMembers = page === "members" && !scope.team;
        const explicitTab = page === "settings" || page === "budget_groups" || organizationMembers;
        return (
          <a
            key={page}
            href={
              page === "usage"
                ? customerHierarchyHref("usage", scope)
                : `${customerHierarchyHref("manage", scope)}${explicitTab ? `&tab=${page}` : ""}`
            }
            aria-current={active === page ? "page" : undefined}
            className={cn(
              "border-b-2 border-transparent px-0.5 pb-3 text-sm font-medium text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring",
              active === page && "border-primary text-foreground",
            )}
          >
            {t(LABELS[page])}
          </a>
        );
      })}
    </nav>
  );
}
