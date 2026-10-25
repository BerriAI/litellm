"use client";

import { englishTranslate, type Translate } from "@/i18n/language";
import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { Copy, KeyRound, Layers, MoreHorizontal, Pencil, Trash2, Users } from "lucide-react";

import { DataTableMultiSortHeader, DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import { orgDetailHref } from "@/utils/entityLinks";
import { copyToClipboard, formatNumberWithCommas } from "@/utils/dataUtils";

import { Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";
import { TeamMemberSpendBudgetCell } from "./TeamMemberSpendBudgetCell";

interface ResourceTone {
  icon: typeof Users;
  className: string;
}

const RESOURCE_TONES: Record<"members" | "models" | "keys", ResourceTone> = {
  members: {
    icon: Users,
    className:
      "bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-950 dark:text-violet-300 dark:ring-violet-400/30",
  },
  models: { icon: Layers, className: "bg-info/10 text-info ring-sky-600/20" },
  keys: { icon: KeyRound, className: "bg-success/10 text-success ring-emerald-600/20" },
};

const teamMemberCount = (team: Team): number => team.members_count ?? team.members_with_roles?.length ?? 0;
const teamModelCount = (team: Team): number => team.models?.length ?? 0;
const teamKeyCount = (team: Team): number => team.keys_count ?? team.keys?.length ?? 0;

function ResourcesCell({ team }: { team: Team }) {
  const { t } = useLanguage();
  const items = [
    { key: "members" as const, label: t("members"), count: teamMemberCount(team) },
    { key: "models" as const, label: t("models"), count: teamModelCount(team) },
    { key: "keys" as const, label: t("keys"), count: teamKeyCount(team) },
  ];

  return (
    <div className="flex items-center gap-1.5">
      {items.map((item) => {
        const tone = RESOURCE_TONES[item.key];
        const Icon = tone.icon;
        return (
          <span
            key={item.key}
            title={`${item.count} ${item.label}`}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium ring-1 ring-inset [&_svg]:size-3.5",
              tone.className,
            )}
          >
            <Icon />
            <span className="tabular-nums">{item.count}</span>
          </span>
        );
      })}
    </div>
  );
}

function RateLimitLine({ label, value }: { label: string; value: number | null }) {
  const { t } = useLanguage();
  return (
    <div>
      <span className="text-[10px] font-semibold text-muted-foreground">{label} </span>
      <span className="tabular-nums">{value != null ? formatNumberWithCommas(value) : t("Unlimited")}</span>
    </div>
  );
}

interface TeamRowActionsProps {
  team: Team;
  canManage: boolean;
  onEditTeam: (team: Team) => void;
  onDeleteTeam: (team: Team) => void;
}

function TeamRowActions({ team, canManage, onEditTeam, onDeleteTeam }: TeamRowActionsProps) {
  const { t } = useLanguage();
  const handleCopy = () => {
    void copyToClipboard(team.team_id, t("Team ID copied"));
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("Open team actions")}
        data-testid={`team-actions-${team.team_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        {canManage && (
          <DropdownMenuItem onClick={() => onEditTeam(team)} data-testid="team-action-edit">
            <Pencil />
            {t("Edit team")}
          </DropdownMenuItem>
        )}
        <DropdownMenuItem onClick={handleCopy} data-testid="team-action-copy">
          <Copy />
          {t("Copy team ID")}
        </DropdownMenuItem>
        {canManage && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onClick={() => onDeleteTeam(team)} data-testid="team-action-delete">
              <Trash2 />
              {t("Delete team")}
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface TeamTableColumnsDeps {
  t?: Translate;
  organizations: Organization[];
  userRole: string | null;
  onSelectTeam: (team: Team) => void;
  onEditTeam: (team: Team) => void;
  onDeleteTeam: (team: Team) => void;
}

export const getTeamTableColumns = ({
  t = englishTranslate,
  organizations,
  userRole,
  onSelectTeam,
  onEditTeam,
  onDeleteTeam,
}: TeamTableColumnsDeps): ColumnDef<Team>[] => {
  const canManage = userRole === "Admin";

  return [
    {
      id: "team_alias",
      accessorKey: "team_alias",
      meta: {
        title: t("Team"),
        renderSkeleton: () => (
          <div className="flex flex-col gap-2 py-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3.5 w-24 opacity-65" />
          </div>
        ),
      },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Team")} variant="header-cycle" />,
      size: 260,
      enableSorting: true,
      cell: ({ row }) => {
        const team = row.original;
        const hasAlias = Boolean(team.team_alias);
        return (
          <IdentityCell
            title={team.team_alias || team.team_id}
            subtitle={hasAlias ? team.team_id : undefined}
            onClick={() => onSelectTeam(team)}
          />
        );
      },
    },
    {
      id: "organization_alias",
      accessorKey: "organization_id",
      meta: { title: t("Organization") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Organization")} />,
      sortDescFirst: false,
      size: 160,
      enableSorting: true,
      cell: (info) => {
        const orgId = info.getValue() as string | null;
        if (!orgId) return <span className="text-muted-foreground">—</span>;
        const org = organizations.find((o) => o.organization_id === orgId);
        const displayValue = org?.organization_alias || orgId;
        const width = info.cell.column.getSize();
        return (
          <span className="block" style={{ maxWidth: width }} title={displayValue}>
            <IdentityCell title={displayValue} titleClassName="text-sm font-normal" href={orgDetailHref(orgId)} />
          </span>
        );
      },
    },
    {
      id: "resources",
      meta: {
        title: t("Resources"),
        renderSkeleton: () => (
          <div className="flex items-center gap-1.5">
            <Skeleton className="h-6 w-12 rounded-md" />
            <Skeleton className="h-6 w-12 rounded-md" />
            <Skeleton className="h-6 w-12 rounded-md opacity-65" />
          </div>
        ),
      },
      header: ({ table }) => (
        <DataTableMultiSortHeader
          table={table}
          fields={[
            { id: "members", label: t("Members") },
            { id: "models", label: t("Models") },
            { id: "keys_count", label: t("Keys") },
          ]}
        />
      ),
      size: 210,
      enableSorting: false,
      cell: ({ row }) => <ResourcesCell team={row.original} />,
    },
    {
      id: "spend",
      accessorKey: "spend",
      meta: { title: t("Spend / Budget"), skeleton: "meter", numeric: true },
      header: ({ table }) => (
        <DataTableMultiSortHeader
          table={table}
          fields={[
            { id: "spend", label: t("Spend") },
            { id: "max_budget", label: t("Budget") },
          ]}
        />
      ),
      size: 200,
      enableSorting: false,
      cell: ({ row }) => (
        <TeamMemberSpendBudgetCell
          teamSpend={row.original.spend}
          teamMaxBudget={row.original.max_budget}
          callerMembership={row.original.caller_membership}
        />
      ),
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      meta: { title: t("Created") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Created")} variant="header-cycle" />,
      size: 130,
      enableSorting: true,
      cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" />,
    },
    {
      id: "members",
      accessorFn: teamMemberCount,
      meta: { title: t("Members"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Members")} />,
      sortDescFirst: true,
      size: 110,
      enableSorting: true,
      cell: ({ row }) => <span className="text-sm tabular-nums">{teamMemberCount(row.original)}</span>,
    },
    {
      id: "models",
      accessorFn: teamModelCount,
      meta: { title: t("Models"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Models")} />,
      sortDescFirst: true,
      size: 100,
      enableSorting: true,
      cell: ({ row }) => <span className="text-sm tabular-nums">{teamModelCount(row.original)}</span>,
    },
    {
      id: "rate_limits",
      meta: { title: t("Rate Limits"), skeleton: "twoLine" },
      header: ({ table }) => (
        <DataTableMultiSortHeader
          table={table}
          fields={[
            { id: "tpm_limit", label: "TPM" },
            { id: "rpm_limit", label: "RPM" },
          ]}
        />
      ),
      size: 140,
      enableSorting: false,
      cell: ({ row }) => (
        <div className="text-xs leading-tight">
          <RateLimitLine label="TPM" value={row.original.tpm_limit} />
          <RateLimitLine label="RPM" value={row.original.rpm_limit} />
        </div>
      ),
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      meta: { title: t("Updated") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Updated")} />,
      sortDescFirst: false,
      size: 130,
      enableSorting: true,
      cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" fallback={t("Never")} />,
    },
    {
      id: "actions",
      meta: { className: "text-right", headerClassName: "text-right" },
      header: () => <span className="sr-only">{t("Actions")}</span>,
      size: 60,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <TeamRowActions
            team={row.original}
            canManage={canManage}
            onEditTeam={onEditTeam}
            onDeleteTeam={onDeleteTeam}
          />
        </div>
      ),
    },
  ];
};

export const TEAM_TABLE_HIDDEN_COLUMNS: Record<string, boolean> = {
  members: false,
  models: false,
  rate_limits: false,
  updated_at: false,
};
