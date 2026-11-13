"use client";

import { ChevronRight, Pencil } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import CustomerHierarchyNavigation, { customerHierarchyHref } from "./CustomerHierarchyNavigation";
import CustomerHierarchySettings from "./CustomerHierarchySettings";
import { customerEntityId, customerEntityLabel } from "./customerHierarchyForms";
import useCustomerHierarchy, { type CustomerHierarchyContext } from "./useCustomerHierarchy";
import type { CustomerHierarchyScope } from "./customerHierarchyApi";

type Hierarchy = ReturnType<typeof useCustomerHierarchy>;
interface ScopeProps {
  hierarchy: Hierarchy;
  selected: CustomerHierarchyContext;
  scope: CustomerHierarchyScope;
}

function HierarchyHeader({ hierarchy, selected }: Pick<ScopeProps, "hierarchy" | "selected">) {
  const { t } = useTranslation();
  const entity = selected.entity;
  const team = selected.team;
  const label = customerEntityLabel(entity);
  const kindLabel = {
    organization: t("customerPortal.hierarchy.organization"),
    team: t("customerPortal.hierarchy.team"),
    member: t("customerPortal.hierarchy.member"),
  }[entity.kind];
  return (
    <header>
      <nav
        aria-label={t("customerPortal.analytics.breadcrumb")}
        className="mb-5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
      >
        <a href={uiHref("customers")} className="hover:text-foreground">
          {t("customerPortal.analytics.customers")}
        </a>
        <ChevronRight aria-hidden="true" className="size-3" />
        <button
          type="button"
          onClick={hierarchy.openOrganization}
          className="hover:text-foreground"
          aria-current={!team ? "page" : undefined}
        >
          {selected.organization.alias || selected.organization.customer_org_id}
        </button>
        {team && (
          <>
            <ChevronRight aria-hidden="true" className="size-3" />
            <button
              type="button"
              onClick={() => hierarchy.openTeam(team.customer_team_id)}
              className="hover:text-foreground"
              aria-current={!selected.member ? "page" : undefined}
            >
              {team.alias || team.customer_team_id}
            </button>
          </>
        )}
        {selected.member && (
          <>
            <ChevronRight aria-hidden="true" className="size-3" />
            <span aria-current="page">{selected.member.alias || selected.member.customer_member_id}</span>
          </>
        )}
      </nav>
      <p className="mb-1 text-xs text-muted-foreground">{kindLabel}</p>
      <h1 className="text-2xl font-semibold tracking-tight">{label}</h1>
      <p className="mt-1.5 text-sm text-muted-foreground">{t("customerPortal.hierarchy.subtitle")}</p>
      <p className="mt-3 font-mono text-xs text-muted-foreground">{customerEntityId(entity)}</p>
    </header>
  );
}

function hierarchyRows({ hierarchy, selected, scope }: ScopeProps) {
  const list = hierarchy.children.data;
  if (list?.kind === "teams")
    return list.data.map((team) => ({
      id: team.customer_team_id,
      label: team.alias || team.customer_team_id,
      spend: team.spend,
      budget: team.budget?.max_budget,
      open: () => hierarchy.openTeam(team.customer_team_id),
      edit: () => hierarchy.openTeam(team.customer_team_id, true),
      usageScope: { org: scope.org, team: team.customer_team_id, member: null },
    }));
  if (list?.kind === "members")
    return list.data.map((member) => ({
      id: member.customer_member_id,
      label: member.alias || member.customer_member_id,
      spend: member.spend,
      budget: member.effective_budget.max_budget,
      open: () => hierarchy.openMember(member.customer_member_id),
      edit: () => hierarchy.openMember(member.customer_member_id),
      usageScope: { org: scope.org, team: selected.team?.customer_team_id ?? null, member: member.customer_member_id },
    }));
  return [];
}

function HierarchyChildTable(props: ScopeProps) {
  const { t } = useTranslation();
  const { hierarchy } = props;
  const { children, childTab, page } = hierarchy;
  const childLabel = childTab === "teams" ? t("customerPortal.hierarchy.teams") : t("customerPortal.hierarchy.members");
  if (children.isError)
    return (
      <ErrorState
        title={t(
          childTab === "teams"
            ? "customerPortal.hierarchy.teamsLoadError"
            : "customerPortal.hierarchy.membersLoadError",
        )}
        message={extractProxyErrorMessage(children.error)}
        onRetry={() => void children.refetch()}
      />
    );
  if (children.isPending)
    return (
      <p role="status" className="py-10 text-sm text-muted-foreground">
        {t("customerPortal.hierarchy.loading")}
      </p>
    );
  const rows = hierarchyRows(props);
  const pages = Math.max(1, children.data.meta.total_pages);
  return (
    <>
      {" "}
      <div className="mb-5">
        <h2 className="text-base font-medium tracking-tight">{childLabel}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {t(childTab === "teams" ? "customerPortal.hierarchy.teamsHint" : "customerPortal.hierarchy.membersHint")}
        </p>
      </div>
      <Table aria-label={childLabel}>
        <TableHeader>
          <TableRow>
            <TableHead className="pl-0">{t("customerPortal.hierarchy.name")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.hierarchy.spend")}</TableHead>
            <TableHead className="text-right">{t("customerPortal.hierarchy.budget")}</TableHead>
            <TableHead className="pr-0">
              <span className="sr-only">{t("customerPortal.hierarchy.settings")}</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.id}>
              <TableCell className="py-4 pl-0">
                <button type="button" onClick={row.open} className="block text-left font-medium hover:text-primary">
                  {row.label}
                </button>
                <span className="mt-1 block font-mono text-[11px] text-muted-foreground">{row.id}</span>
              </TableCell>
              <TableCell className="text-right tabular-nums">{formatUsd(row.spend)}</TableCell>
              <TableCell className="text-right tabular-nums">
                {row.budget == null ? t("customerPortal.hierarchy.noBudget") : formatUsd(row.budget)}
              </TableCell>
              <TableCell className="pr-0 text-right">
                <div className="flex justify-end gap-2">
                  <a
                    href={customerHierarchyHref("usage", row.usageScope)}
                    className="inline-flex h-8 items-center px-2 text-xs text-muted-foreground hover:text-foreground"
                  >
                    {t("customerPortal.hierarchy.viewUsage")}
                  </a>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={row.edit}
                    aria-label={`${t("customerPortal.hierarchy.edit")} ${row.label}`}
                  >
                    <Pencil aria-hidden="true" className="size-3.5" />
                    {t("customerPortal.hierarchy.edit")}
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} className="py-12 text-center text-sm text-muted-foreground">
                {t(
                  childTab === "teams"
                    ? "customerPortal.hierarchy.emptyTeams"
                    : "customerPortal.hierarchy.emptyMembers",
                )}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {pages > 1 && (
        <div className="mt-4 flex items-center justify-between border-t pt-4 text-xs text-muted-foreground">
          <span>{t("customerPortal.hierarchy.pagination", { page, pages })}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => hierarchy.setPage(page - 1)}>
              {t("customerPortal.hierarchy.previous")}
            </Button>
            <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => hierarchy.setPage(page + 1)}>
              {t("customerPortal.hierarchy.next")}
            </Button>
          </div>
        </div>
      )}
    </>
  );
}

export default function CustomerHierarchyManager() {
  const { t } = useTranslation();
  const router = useRouter();
  const hierarchy = useCustomerHierarchy();
  const { scope, context, selected, tab } = hierarchy;

  if (hierarchy.isLoading)
    return (
      <p role="status" className="px-8 py-12 text-sm text-muted-foreground">
        {t("customerPortal.hierarchy.loading")}
      </p>
    );
  if (!hierarchy.canManage) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
  if (!scope)
    return (
      <div className="space-y-3 px-8 py-12">
        <h1 className="text-xl font-semibold">{t("customerPortal.hierarchy.title")}</h1>
        <p className="text-sm text-muted-foreground">{t("customerPortal.hierarchy.missingOrganization")}</p>
        <a href={uiHref("customers")} className="text-sm text-primary hover:underline">
          {t("customerPortal.hierarchy.backToCustomers")}
        </a>
      </div>
    );
  if (context.isError)
    return (
      <ErrorState
        title={t("customerPortal.hierarchy.error")}
        message={extractProxyErrorMessage(context.error)}
        onRetry={() => void context.refetch()}
      />
    );
  if (!selected)
    return (
      <p role="status" className="px-8 py-12 text-sm text-muted-foreground">
        {t("customerPortal.hierarchy.loading")}
      </p>
    );

  const entity = selected.entity;
  const currentScope = {
    org: scope.org,
    team: selected.team?.customer_team_id ?? null,
    member: selected.member?.customer_member_id ?? null,
  };
  const onDeleted = () => {
    hierarchy.refresh();
    if (selected.member && selected.team) hierarchy.openTeam(selected.team.customer_team_id);
    else if (selected.team) hierarchy.openOrganization();
    else router.push(uiHref("customers"));
  };

  return (
    <main
      className="mx-auto w-full max-w-[1600px] space-y-7 px-5 py-6 sm:px-8"
      aria-label={t("customerPortal.hierarchy.title")}
    >
      <HierarchyHeader hierarchy={hierarchy} selected={selected} />
      <CustomerHierarchyNavigation scope={currentScope} active={tab} />
      <section>
        {tab === "settings" ? (
          <CustomerHierarchySettings
            key={customerEntityId(entity)}
            organization={selected.organization}
            entity={entity}
            onSaved={hierarchy.refresh}
            onDeleted={onDeleted}
          />
        ) : (
          <HierarchyChildTable hierarchy={hierarchy} selected={selected} scope={scope} />
        )}
      </section>
    </main>
  );
}
