import { useTranslation } from "react-i18next";
import { cn } from "@/lib/cva.config";
import { uiHref } from "@/utils/uiHref";
import type { CustomerUsageScope } from "./customerUsageApi";

type CustomerHierarchyView = "usage" | "teams" | "members" | "settings";
const LABELS = {
  usage: "customerPortal.hierarchy.usage",
  teams: "customerPortal.hierarchy.teams",
  members: "customerPortal.hierarchy.members",
  settings: "customerPortal.hierarchy.settings",
} as const;

export function customerHierarchyHref(page: "usage" | "manage", scope: CustomerUsageScope): string {
  const query = new URLSearchParams({ org: scope.org });
  if (scope.team) query.set("team", scope.team);
  if (scope.member) query.set("member", scope.member);
  return `${uiHref(`customers/${page}`)}?${query}`;
}

export default function CustomerHierarchyNavigation({
  scope,
  active,
}: {
  scope: CustomerUsageScope;
  active: CustomerHierarchyView;
}) {
  const { t } = useTranslation();
  const pages: readonly CustomerHierarchyView[] = scope.member
    ? ["usage", "settings"]
    : ["usage", scope.team ? "members" : "teams", "settings"];
  return (
    <nav aria-label={t("customerPortal.hierarchy.customerNavigation")} className="flex gap-6 border-b">
      {pages.map((page) => (
        <a
          key={page}
          href={
            page === "usage"
              ? customerHierarchyHref("usage", scope)
              : `${customerHierarchyHref("manage", scope)}${page === "settings" ? "&tab=settings" : ""}`
          }
          aria-current={active === page ? "page" : undefined}
          className={cn(
            "border-b-2 border-transparent px-0.5 pb-3 text-sm font-medium text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring",
            active === page && "border-primary text-foreground",
          )}
        >
          {t(LABELS[page])}
        </a>
      ))}
    </nav>
  );
}
