"use client";

import { useState } from "react";
import { MoreHorizontal, Pencil, Plus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import MoveCustomerMemberDialog from "./MoveCustomerMemberDialog";
import type { Member } from "./customerHierarchyForms";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { extractProxyErrorMessage } from "@/lib/http/client";
import CustomerHierarchyNavigation, { useCustomerHierarchyLinks } from "./CustomerHierarchyNavigation";
import { CustomerDetailHeader, CustomerDetailLayout } from "./CustomerDetailLayout";
import CustomerHierarchyBreadcrumbs from "./CustomerHierarchyBreadcrumbs";
import CustomerHierarchySettings from "./CustomerHierarchySettings";
import CustomerEntityBudgetDialog from "./CustomerEntityBudgetDialog";
import CustomerBudgetGroups from "./CustomerBudgetGroups";
import CustomerBudgetSummary from "./CustomerBudgetSummary";
import CustomerMemberApps from "../../apps/_components/CustomerMemberApps";
import CustomerMemberSearch from "./CustomerMemberSearch";
import { customerUsageEntity, isCustomerUsageIdentifier, type CustomerUsageEntity } from "./customerUsageApi";
import {
  CustomerSpendCells,
  CustomerSpendHeaders,
  CustomerSpendIdentity,
  CustomerSpendTier,
} from "./CustomerSpendCells";
import CreateCustomerHierarchyDialog from "./CreateCustomerHierarchyDialog";
import { customerEntityId, customerEntityLabel, type CreatedCustomerHierarchyEntity } from "./customerHierarchyForms";
import useCustomerHierarchy, { type CustomerHierarchyContext } from "./useCustomerHierarchy";
import type { CustomerHierarchyScope } from "./customerHierarchyApi";
import { CustomerBudgetLimitCell, CustomerBudgetSortHeader } from "./CustomerBudgetColumn";

type Hierarchy = ReturnType<typeof useCustomerHierarchy>;
interface ScopeProps {
  hierarchy: Hierarchy;
  selected: CustomerHierarchyContext;
  scope: CustomerHierarchyScope;
}

function HierarchyHeader({ hierarchy, selected }: Pick<ScopeProps, "hierarchy" | "selected">) {
  const { customerHierarchyHref } = useCustomerHierarchyLinks();
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
  return (
    <CustomerDetailHeader
      title={label}
      breadcrumbs={
        <CustomerHierarchyBreadcrumbs
          scope={budgetScope}
          page="manage"
          organizationLabel={selected.organization.alias}
          teamLabel={team?.alias}
          memberLabel={selected.member?.alias}
          onOrganizationClick={hierarchy.openOrganization}
          onTeamClick={() => {
            if (team) hierarchy.openTeam(team.customer_team_id);
          }}
        />
      }
      actions={
        <div className="w-full sm:w-72">
          <CustomerBudgetSummary entity={budgetEntity} settingsHref={budgetSettingsHref} scope={budgetScope} />
        </div>
      }
    >
      {label !== customerEntityId(entity) && (
        <p className="mt-2 break-all font-mono text-xs text-muted-foreground">{customerEntityId(entity)}</p>
      )}
    </CustomerDetailHeader>
  );
}

function hierarchyRows({ hierarchy, selected, scope }: ScopeProps) {
  const list = hierarchy.children.data;
  if (list?.kind === "teams")
    return list.data.map((team) => ({
      id: team.customer_team_id,
      label: team.alias || team.customer_team_id,
      entity: customerUsageEntity(team.customer_team_id, team),
      edit: () => hierarchy.openTeam(team.customer_team_id, true),
      member: null,
      parentTeamId: null,
      openParentTeam: () => hierarchy.openTeam(team.customer_team_id),
      usageScope: { org: scope.org, team: team.customer_team_id, member: null },
    }));
  if (list?.kind === "members")
    return list.data.map((member) => ({
      id: member.customer_member_id,
      label: member.alias || member.customer_member_id,
      entity: customerUsageEntity(member.customer_member_id, member, selected.organization.default_budgets),
      edit: () => hierarchy.openMember(member.customer_member_id, member.parent_team_ids[0] ?? ""),
      member,
      parentTeamId: member.parent_team_ids[0] ?? "",
      openParentTeam: () => hierarchy.openTeam(member.parent_team_ids[0] ?? ""),
      usageScope: { org: scope.org, team: selected.team?.customer_team_id ?? null, member: member.customer_member_id },
    }));
  return [];
}

function HierarchyCreateAction({ hierarchy, selected, scope }: ScopeProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const parent =
    hierarchy.childTab === "members"
      ? { kind: "member" as const, team: selected.team ?? undefined }
      : { kind: "team" as const };
  const onCreated = (entity: CreatedCustomerHierarchyEntity) => {
    hierarchy.refresh();
    hierarchy.openCreatedScope(scope, entity);
  };
  return (
    <>
      <Button type="button" onClick={() => setOpen(true)}>
        <Plus aria-hidden="true" className="size-4" />
        {t(parent.kind === "member" ? "customerPortal.hierarchy.createMember" : "customerPortal.hierarchy.createTeam")}
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

function HierarchyChildRows(props: ScopeProps & { childLabel: string; showTeam: boolean }) {
  const { customerHierarchyHref } = useCustomerHierarchyLinks();
  const router = useRouter();
  const [movingMember, setMovingMember] = useState<Member | null>(null);
  const [editingMember, setEditingMember] = useState<string | null>(null);
  const { t } = useTranslation();
  const { hierarchy, childLabel, showTeam } = props;
  const { children, childUsage, childTab, page } = hierarchy;
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
  const emptyScope = showTeam
    ? "customerPortal.hierarchy.emptyOrganizationMembers"
    : "customerPortal.hierarchy.emptyMembers";
  const emptyMembers = hierarchy.memberSearch ? "customerPortal.hierarchy.noMemberMatches" : emptyScope;
  const monthly = new Map(childUsage.data?.monthly.map((row) => [row.id, row]));
  const yearly = new Map(childUsage.data?.yearly.map((row) => [row.id, row.spend]));
  return (
    <>
      {childUsage.isError && (
        <ErrorState message={extractProxyErrorMessage(childUsage.error)} onRetry={() => void childUsage.refetch()} />
      )}
      <Table aria-label={childLabel}>
        <TableHeader>
          <TableRow>
            <TableHead className="pl-0">{t("customerPortal.hierarchy.name")}</TableHead>
            {showTeam && <TableHead>{t("customerPortal.hierarchy.team")}</TableHead>}
            {childTab === "members" && <TableHead>{t("customerPortal.budgetEditor.group")}</TableHead>}
            <CustomerSpendHeaders>
              <CustomerBudgetSortHeader sort={hierarchy.budgetSort} onSort={hierarchy.toggleBudgetSort} />
            </CustomerSpendHeaders>
            <TableHead className="pr-0">
              <span className="sr-only">{t("customerPortal.hierarchy.settings")}</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.id}
              className="cursor-pointer"
              onClick={(event) => {
                if (
                  !(event.target instanceof Element) ||
                  !event.currentTarget.contains(event.target) ||
                  event.target.closest("a, button")
                )
                  return;
                router.push(customerHierarchyHref("usage", row.usageScope));
              }}
            >
              <TableCell className="py-4 pl-0">
                <CustomerSpendIdentity entity={row.entity} href={customerHierarchyHref("usage", row.usageScope)} />
              </TableCell>
              {showTeam && (
                <TableCell>
                  <button
                    type="button"
                    className="block max-w-40 truncate text-xs text-muted-foreground hover:text-primary"
                    title={row.parentTeamId ?? undefined}
                    onClick={row.openParentTeam}
                  >
                    {row.parentTeamId}
                  </button>
                </TableCell>
              )}
              {row.member && <CustomerSpendTier entity={row.entity} />}
              {childUsage.data && isCustomerUsageIdentifier(row.id) ? (
                <CustomerSpendCells
                  entity={row.entity}
                  monthlySpend={monthly.get(row.id)?.spend ?? 0}
                  yearlySpend={yearly.get(row.id) ?? 0}
                  tokens={monthly.get(row.id)?.tokens ?? 0}
                  organizationSpend={childUsage.data.organizationSpend}
                />
              ) : (
                <>
                  <TableCell className="text-center text-xs text-muted-foreground" role="status">
                    {t(
                      childUsage.isError || !isCustomerUsageIdentifier(row.id)
                        ? "customerPortal.analytics.unknownSpend"
                        : "customerPortal.hierarchy.loading",
                    )}
                  </TableCell>
                  <CustomerBudgetLimitCell entity={row.entity} />
                  <TableCell colSpan={4} />
                </>
              )}
              <TableCell className="pr-0 text-right">
                <div className="flex justify-end gap-2">
                  {row.member ? (
                    <DropdownMenu>
                      <DropdownMenuTrigger
                        render={
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label={t("customerPortal.hierarchy.memberActions", { name: row.label })}
                          />
                        }
                      >
                        <MoreHorizontal className="size-4" />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => setEditingMember(row.id)}>
                          {t("customerPortal.analytics.editBudget")}
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={row.edit}>{t("customerPortal.hierarchy.settings")}</DropdownMenuItem>
                        <DropdownMenuItem onClick={() => setMovingMember(row.member)}>
                          {t("customerPortal.hierarchy.moveTeams")}
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  ) : (
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
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
          {rows.length === 0 && (
            <TableRow>
              <TableCell
                colSpan={8 + Number(showTeam) + Number(childTab === "members")}
                className="py-12 text-center text-sm text-muted-foreground"
              >
                {t(childTab === "teams" ? "customerPortal.hierarchy.emptyTeams" : emptyMembers)}
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
      {editingMember && (
        <CustomerEntityBudgetDialog
          memberId={editingMember}
          organizationId={props.scope.org}
          onClose={() => setEditingMember(null)}
        />
      )}
      {movingMember && (
        <MoveCustomerMemberDialog
          member={movingMember}
          organizationId={props.scope.org}
          onClose={() => setMovingMember(null)}
        />
      )}
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

function HierarchyChildTable(props: ScopeProps) {
  const { t } = useTranslation();
  const { hierarchy } = props;
  const { childTab } = hierarchy;
  const showTeam = childTab === "members" && props.selected.team === null;
  const childLabel = childTab === "teams" ? t("customerPortal.hierarchy.teams") : t("customerPortal.hierarchy.members");
  return (
    <>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        {childTab === "members" && (
          <CustomerMemberSearch
            key={props.selected.team?.customer_team_id ?? ""}
            value={hierarchy.memberSearch}
            onSearch={hierarchy.setMemberSearch}
          />
        )}
        <div className="ml-auto">
          <HierarchyCreateAction hierarchy={hierarchy} selected={props.selected} scope={props.scope} />
        </div>
      </div>
      <HierarchyChildRows {...props} childLabel={childLabel} showTeam={showTeam} />
    </>
  );
}

export default function CustomerHierarchyManager() {
  const { customersHref } = useCustomerHierarchyLinks();
  const { t } = useTranslation();
  const router = useRouter();
  const hierarchy = useCustomerHierarchy();
  const { scope, context, selected, tab } = hierarchy;

  if (hierarchy.isLoading)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        {t("customerPortal.hierarchy.loading")}
      </p>
    );
  if (!hierarchy.canManage) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
  if (!scope)
    return (
      <div className="space-y-3">
        <h1 className="text-xl font-semibold">{t("customerPortal.hierarchy.title")}</h1>
        <p className="text-sm text-muted-foreground">{t("customerPortal.hierarchy.missingOrganization")}</p>
        <a href={customersHref} className="text-sm text-primary hover:underline">
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
      <p role="status" className="text-sm text-muted-foreground">
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
    else router.push(customersHref);
  };

  return (
    <CustomerDetailLayout label={t("customerPortal.hierarchy.title")}>
      <HierarchyHeader hierarchy={hierarchy} selected={selected} />
      <CustomerHierarchyNavigation scope={currentScope} active={tab} />
      {selected.member && (
        <CustomerMemberApps
          key={selected.member.customer_member_id}
          memberId={selected.member.customer_member_id}
          organizationId={selected.organization.customer_org_id}
        />
      )}
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
    </CustomerDetailLayout>
  );
}
