"use client";

import { useState } from "react";
import { ChevronRight, Pencil, Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import CustomerHierarchyNavigation, { customerHierarchyHref } from "./CustomerHierarchyNavigation";
import CustomerHierarchySettings from "./CustomerHierarchySettings";
import CustomerBudgetGroups from "./CustomerBudgetGroups";
import CustomerBudgetSummary from "./CustomerBudgetSummary";
import { customerUsageEntity, type CustomerUsageEntity } from "./customerUsageApi";
import { CustomerSpendCells, CustomerSpendHeaders, CustomerSpendIdentity } from "./CustomerSpendCells";
import CreateCustomerHierarchyDialog from "./CreateCustomerHierarchyDialog";
import { customerEntityId, customerEntityLabel, type CreatedCustomerHierarchyEntity } from "./customerHierarchyForms";
import useCustomerHierarchy, { type CustomerHierarchyContext } from "./useCustomerHierarchy";
import type { CustomerHierarchyScope } from "./customerHierarchyApi";

type Hierarchy = ReturnType<typeof useCustomerHierarchy>;
const CHILD_HINTS = {
  teams: "customerPortal.hierarchy.teamsHint",
  members: "customerPortal.hierarchy.membersHint",
} as const;
interface ScopeProps {
  hierarchy: Hierarchy;
  selected: CustomerHierarchyContext;
  scope: CustomerHierarchyScope;
}

const DESCRIPTION_KEYS = {
  organization: "customerPortal.hierarchy.subtitle",
  team: "customerPortal.hierarchy.teamSubtitle",
  member: "customerPortal.hierarchy.memberSubtitle",
} as const;

function HierarchyHeader({ hierarchy, selected }: Pick<ScopeProps, "hierarchy" | "selected">) {
  const { t } = useTranslation();
  const entity = selected.entity;
  const team = selected.team;
  const label = customerEntityLabel(entity);
  const budget = entity.kind === "organization" ? entity.record.budget : entity.record.effective_budget;
  const budgetEntity: CustomerUsageEntity = {
    id: customerEntityId(entity),
    label,
    spend: entity.record.spend,
    budgetLimit: budget?.max_budget ?? null,
    budgetDuration: budget?.budget_duration ?? null,
    budgetLimits: budget?.budget_limits ?? [],
    budgetResetAt: entity.record.budget_reset_at ?? null,
    budgetUsage: entity.record.budget_usage,
  };
  const budgetScope = {
    org: selected.organization.customer_org_id,
    team: selected.team?.customer_team_id ?? null,
    member: selected.member?.customer_member_id ?? null,
  };
  const budgetSettingsHref = `${customerHierarchyHref("manage", budgetScope)}&tab=settings`;
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
          {t("customerPortal.entities.organizations")}
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
      <div className="flex flex-wrap items-start justify-between gap-6">
        <div className="min-w-0 flex-1 basis-72">
          <p className="mb-1 text-xs text-muted-foreground">{kindLabel}</p>
          <h1 className="text-2xl font-semibold tracking-tight">{label}</h1>
          <p className="mt-1.5 text-sm text-muted-foreground">{t(DESCRIPTION_KEYS[entity.kind])}</p>
          <p className="mt-3 font-mono text-xs text-muted-foreground">{customerEntityId(entity)}</p>
        </div>
        <div className="w-full sm:w-72">
          <CustomerBudgetSummary entity={budgetEntity} settingsHref={budgetSettingsHref} />
        </div>
      </div>
    </header>
  );
}

function hierarchyRows({ hierarchy, selected, scope }: ScopeProps) {
  const list = hierarchy.children.data;
  if (list?.kind === "teams")
    return list.data.map((team) => ({
      id: team.customer_team_id,
      label: team.alias || team.customer_team_id,
      entity: customerUsageEntity(team.customer_team_id, team),
      open: () => hierarchy.openTeam(team.customer_team_id),
      edit: () => hierarchy.openTeam(team.customer_team_id, true),
      parentTeamId: null,
      openParentTeam: () => hierarchy.openTeam(team.customer_team_id),
      usageScope: { org: scope.org, team: team.customer_team_id, member: null },
    }));
  if (list?.kind === "members")
    return list.data.map((member) => ({
      id: member.customer_member_id,
      label: member.alias || member.customer_member_id,
      entity: customerUsageEntity(member.customer_member_id, member),
      open: () => hierarchy.openMember(member.customer_member_id, member.parent_team_ids[0] ?? ""),
      edit: () => hierarchy.openMember(member.customer_member_id, member.parent_team_ids[0] ?? ""),
      parentTeamId: member.parent_team_ids[0] ?? "",
      openParentTeam: () => hierarchy.openTeam(member.parent_team_ids[0] ?? ""),
      usageScope: { org: scope.org, team: selected.team?.customer_team_id ?? null, member: member.customer_member_id },
    }));
  return [];
}

function HierarchyCreateAction({ hierarchy, selected, scope }: ScopeProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  if (hierarchy.childTab === "members" && selected.team === null)
    return (
      <Button variant="outline" render={<a href={`${customerHierarchyHref("manage", scope)}&tab=teams`} />}>
        {t("customerPortal.hierarchy.chooseTeam")}
      </Button>
    );
  const parent = selected.team ? { kind: "member" as const, team: selected.team } : { kind: "team" as const };
  const onCreated = (entity: CreatedCustomerHierarchyEntity) => {
    hierarchy.refresh();
    hierarchy.openCreatedScope(scope, entity);
  };
  return (
    <>
      <Button type="button" onClick={() => setOpen(true)}>
        <Plus aria-hidden="true" className="size-4" />
        {t(selected.team ? "customerPortal.hierarchy.createMember" : "customerPortal.hierarchy.createTeam")}
      </Button>
      <CreateCustomerHierarchyDialog
        {...parent}
        open={open}
        onOpenChange={setOpen}
        organization={selected.organization}
        onCreated={onCreated}
      />
    </>
  );
}

function HierarchyChildTable(props: ScopeProps) {
  const { t } = useTranslation();
  const { hierarchy } = props;
  const { children, childUsage, childTab, page } = hierarchy;
  const showTeam = childTab === "members" && props.selected.team === null;
  const hint = showTeam ? "customerPortal.hierarchy.organizationMembersHint" : CHILD_HINTS[childTab];
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
  const monthly = new Map(childUsage.data?.monthly.map((row) => [row.id, row]));
  const yearly = new Map(childUsage.data?.yearly.map((row) => [row.id, row.spend]));
  return (
    <>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-medium tracking-tight">{childLabel}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{t(hint)}</p>
        </div>
        <HierarchyCreateAction hierarchy={hierarchy} selected={props.selected} scope={props.scope} />
      </div>
      {childUsage.isError && (
        <ErrorState message={extractProxyErrorMessage(childUsage.error)} onRetry={() => void childUsage.refetch()} />
      )}
      <p className="mb-4 text-xs text-muted-foreground">{t("customerPortal.analytics.memberSpendHint")}</p>
      <Table aria-label={childLabel}>
        <TableHeader>
          <TableRow>
            <TableHead className="pl-0">{t("customerPortal.hierarchy.name")}</TableHead>
            {showTeam && <TableHead>{t("customerPortal.hierarchy.team")}</TableHead>}
            <CustomerSpendHeaders />
            <TableHead className="pr-0">
              <span className="sr-only">{t("customerPortal.hierarchy.settings")}</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow key={row.id}>
              <TableCell className="py-4 pl-0">
                <CustomerSpendIdentity entity={row.entity} onClick={row.open} />
              </TableCell>
              {showTeam && (
                <TableCell>
                  <button
                    type="button"
                    className="text-xs text-muted-foreground hover:text-primary"
                    onClick={row.openParentTeam}
                  >
                    {row.parentTeamId}
                  </button>
                </TableCell>
              )}
              {childUsage.data ? (
                <CustomerSpendCells
                  entity={row.entity}
                  monthlySpend={monthly.get(row.id)?.spend ?? 0}
                  yearlySpend={yearly.get(row.id) ?? 0}
                  tokens={monthly.get(row.id)?.tokens ?? 0}
                  organizationSpend={childUsage.data.organizationSpend}
                />
              ) : (
                <TableCell colSpan={5} className="text-center text-xs text-muted-foreground" role="status">
                  {t(childUsage.isError ? "customerPortal.analytics.unknownSpend" : "customerPortal.hierarchy.loading")}
                </TableCell>
              )}
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
              <TableCell colSpan={showTeam ? 8 : 7} className="py-12 text-center text-sm text-muted-foreground">
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
        {tab === "budget_groups" ? (
          <CustomerBudgetGroups
            key={scope.org}
            organization={selected.organization}
            onSaved={hierarchy.organizationSaved}
          />
        ) : null}
        {tab === "settings" && (
          <CustomerHierarchySettings
            key={customerEntityId(entity)}
            organization={selected.organization}
            entity={entity}
            onSaved={hierarchy.refresh}
            onDeleted={onDeleted}
          />
        )}
        {(tab === "teams" || tab === "members") && (
          <HierarchyChildTable hierarchy={hierarchy} selected={selected} scope={scope} />
        )}
      </section>
    </main>
  );
}
