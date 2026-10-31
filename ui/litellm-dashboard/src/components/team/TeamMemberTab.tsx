import { useResetTeamMemberBudget } from "@/app/(dashboard)/hooks/teams/useResetTeamMemberBudget";
import { useResetTeamMemberSpend } from "@/app/(dashboard)/hooks/teams/useResetTeamMemberSpend";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { SimpleTooltip } from "@/components/ui/tooltip";
import MemberTable from "@/components/common_components/MemberTable";
import type { StoredModelMaxBudget } from "@/components/key_team_helpers/modelMaxBudgetPayload";
import { Member } from "@/components/networking";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { MoneyCell } from "@/components/shared/table_cells";
import { toast } from "@/lib/toast";
import { describeBudgetResetSchedule } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import type { ParseKeys, TFunction } from "i18next";
import { CircleHelp } from "lucide-react";
import { useMemo, useState, type ComponentProps } from "react";
import { Trans, useTranslation } from "react-i18next";
import { TeamData, TeamMemberBudgetSource, TeamMembership } from "./TeamInfo";

const BUDGET_SOURCE_LABEL_KEYS: Record<Exclude<TeamMemberBudgetSource, "none">, ParseKeys> = {
  team_default: "teams.memberTab.budgetSource.teamDefault",
  custom: "teams.memberTab.budgetSource.custom",
};

const ESCAPED_VALUES = { interpolation: { escapeValue: true } } as const;

const formatBudget = (t: TFunction, value: number | null): string =>
  value === null ? t("teams.memberTab.unlimited") : `$${formatNumberWithCommas(value, 2)}`;

const hasModelBudgets = (budget: StoredModelMaxBudget | null | undefined): budget is StoredModelMaxBudget =>
  budget != null && Object.keys(budget).length > 0;

export const seedMemberBudgetFields = (
  record: Member,
  budget: TeamMembership["litellm_budget_table"] | undefined,
  teamDefaultModelMaxBudget?: StoredModelMaxBudget | null,
): Member => ({
  ...record,
  max_budget_in_team: budget?.max_budget ?? null,
  tpm_limit: budget?.tpm_limit ?? null,
  rpm_limit: budget?.rpm_limit ?? null,
  budget_duration: budget?.budget_duration || null,
  allowed_models: budget?.allowed_models || [],
  temp_budget_increase: budget?.temp_budget_increase ?? null,
  temp_budget_expiry: budget?.temp_budget_expiry ?? null,
  model_max_budget: hasModelBudgets(budget?.model_max_budget)
    ? budget.model_max_budget
    : teamDefaultModelMaxBudget ?? null,
});

interface TeamMemberTabProps {
  teamData: TeamData;
  canEditTeam: boolean;
  handleMemberDelete: (member: Member) => void;
  setSelectedEditMember: (member: Member) => void;
  setIsEditMemberModalVisible: (visible: boolean) => void;
  setIsAddMemberModalVisible: (visible: boolean) => void;
  onMemberSpendReset: () => void;
  onMemberBudgetReset: () => void;
}

export default function TeamMemberTab({
  teamData,
  canEditTeam,
  handleMemberDelete,
  setSelectedEditMember,
  setIsEditMemberModalVisible,
  setIsAddMemberModalVisible,
  onMemberSpendReset,
  onMemberBudgetReset,
}: TeamMemberTabProps) {
  const { t } = useTranslation();
  const [memberToResetSpend, setMemberToResetSpend] = useState<Member | null>(null);
  const [memberToResetBudget, setMemberToResetBudget] = useState<Member | null>(null);
  const { mutate: resetMemberSpend, isPending: isResettingSpend } = useResetTeamMemberSpend();
  const { mutate: resetMemberBudget, isPending: isResettingBudget } = useResetTeamMemberBudget();
  const teamDefaultBudget = teamData.team_info.team_member_budget_table?.max_budget ?? null;

  const membershipsByUserId = useMemo(
    () => new Map(teamData.team_memberships.map((membership) => [membership.user_id, membership])),
    [teamData.team_memberships],
  );

  const findMembership = (userId: string | null): TeamMembership | undefined =>
    userId ? membershipsByUserId.get(userId) : undefined;

  const formatNumber = (value: number | null): string => {
    if (value === null || value === undefined) return "0";

    if (typeof value === "number") {
      // Convert scientific notation to normal decimal
      const normalNumber = Number(value);

      // If it's a whole number, return it without decimals
      if (normalNumber === Math.floor(normalNumber)) {
        return normalNumber.toString();
      }

      // For decimal numbers, use toFixed and remove trailing zeros
      return formatNumberWithCommas(normalNumber, 8).replace(/\.?0+$/, "");
    }

    return "0";
  };

  const getUserCurrentPeriodSpend = (userId: string | null): number => {
    if (!userId) return 0;
    return findMembership(userId)?.spend ?? 0;
  };

  const getUserTotalSpend = (userId: string | null): number => {
    if (!userId) return 0;
    return findMembership(userId)?.total_spend ?? 0;
  };

  const getUserBudgetSource = (userId: string | null): TeamMemberBudgetSource => {
    if (!userId) return "none";
    return findMembership(userId)?.budget_source ?? "none";
  };

  const getUserBudget = (userId: string | null): number | null => {
    if (!userId) return null;
    return findMembership(userId)?.litellm_budget_table?.max_budget ?? teamDefaultBudget;
  };

  // Helper function to get rate limits for a user
  const getUserRateLimits = (userId: string | null): string => {
    if (!userId) return t("teams.memberTab.rateLimits.none");
    const membership = findMembership(userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit != null ? t("teams.memberTab.rateLimits.rpm", { value: formatNumber(rpmLimit) }) : null;
    const tpmText = tpmLimit != null ? t("teams.memberTab.rateLimits.tpm", { value: formatNumber(tpmLimit) }) : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : t("teams.memberTab.rateLimits.none");
  };

  const { data: uiSettingsData } = useUISettings();
  const { userId, userRole } = useAuthorized();
  const disableTeamAdminDeleteTeamUser = Boolean(uiSettingsData?.values?.disable_team_admin_delete_team_user);
  const isUserTeamAdmin = isUserTeamAdminForSingleTeam(teamData.team_info.members_with_roles, userId || "");
  const isProxyAdmin = isProxyAdminRole(userRole || "");

  const getUserAllowedModels = (userId: string | null): string[] | null => {
    if (!userId) return null;
    const models = findMembership(userId)?.litellm_budget_table?.allowed_models;
    return models && models.length > 0 ? models : null;
  };

  const getUserBudgetReset = (userId: string | null): string | null => {
    if (!userId) return null;
    return findMembership(userId)?.litellm_budget_table?.budget_reset_at ?? null;
  };

  const getUserBudgetSchedule = (userId: string | null) => {
    const budget = findMembership(userId)?.litellm_budget_table;
    return describeBudgetResetSchedule(budget?.budget_duration ?? null, budget?.budget_reset_at ?? null);
  };

  const renderBudgetPeriod = (userId: string | null) => {
    const schedule = getUserBudgetSchedule(userId);
    return (
      <SimpleTooltip content={schedule.tooltip}>
        <span>{schedule.period ?? schedule.label}</span>
      </SimpleTooltip>
    );
  };

  const extraColumns: NonNullable<ComponentProps<typeof MemberTable>["extraColumns"]> = [
    {
      title: (
        <span className="flex items-center gap-1">
          {t("teams.memberTab.columns.modelScope.title")}
          <SimpleTooltip content={t("teams.memberTab.columns.modelScope.tooltip")}>
            <CircleHelp className="size-4" aria-label={t("teams.memberTab.columns.modelScope.ariaLabel")} />
          </SimpleTooltip>
        </span>
      ),
      key: "model_scope",
      render: (record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return <span className="text-muted-foreground">{t("teams.memberTab.columns.modelScope.allTeamModels")}</span>;
        }
        const displayed = models.slice(0, 2);
        const remaining = models.length - displayed.length;
        return (
          <div className="flex flex-wrap gap-1">
            {displayed.map((m) => (
              <code key={m} className="rounded bg-muted px-1 py-0.5 text-xs">
                {m}
              </code>
            ))}
            {remaining > 0 && (
              <SimpleTooltip content={models.slice(2).join(", ")}>
                <span className="text-muted-foreground">
                  {t("teams.memberTab.columns.modelScope.moreModels", { count: remaining })}
                </span>
              </SimpleTooltip>
            )}
          </div>
        );
      },
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("teams.memberTab.columns.currentPeriodSpend.title")}
          <SimpleTooltip content={t("teams.memberTab.columns.currentPeriodSpend.tooltip")}>
            <CircleHelp className="size-4" aria-label={t("teams.memberTab.columns.currentPeriodSpend.ariaLabel")} />
          </SimpleTooltip>
        </span>
      ),
      key: "spend",
      numeric: true,
      sortValue: (record: Member) => getUserCurrentPeriodSpend(record.user_id),
      render: (record: Member) => <MoneyCell value={getUserCurrentPeriodSpend(record.user_id)} decimals={2} />,
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("teams.memberTab.columns.totalSpend.title")}
          <SimpleTooltip content={t("teams.memberTab.columns.totalSpend.tooltip")}>
            <CircleHelp className="size-4" aria-label={t("teams.memberTab.columns.totalSpend.ariaLabel")} />
          </SimpleTooltip>
        </span>
      ),
      key: "total_spend",
      numeric: true,
      sortValue: (record: Member) => getUserTotalSpend(record.user_id),
      render: (record: Member) => <MoneyCell value={getUserTotalSpend(record.user_id)} decimals={2} />,
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("teams.memberTab.columns.memberBudget.title")}
          <SimpleTooltip content={t("teams.memberTab.columns.memberBudget.tooltip")}>
            <CircleHelp className="size-4" aria-label={t("teams.memberTab.columns.memberBudget.ariaLabel")} />
          </SimpleTooltip>
        </span>
      ),
      key: "budget",
      numeric: true,
      sortValue: (record: Member) => getUserBudget(record.user_id),
      render: (record: Member) => {
        const source = getUserBudgetSource(record.user_id);
        return (
          <span className="flex items-center justify-end gap-2">
            <MoneyCell
              value={getUserBudget(record.user_id)}
              decimals={2}
              emptyText={t("teams.memberTab.unlimited")}
              showZero
            />
            {source !== "none" && (
              <Badge variant={source === "custom" ? "outline" : "secondary"} data-testid="member-budget-source">
                {t(BUDGET_SOURCE_LABEL_KEYS[source])}
              </Badge>
            )}
            {source === "custom" && canEditTeam && (
              <Button
                variant="link"
                size="xs"
                className="h-auto p-0"
                data-testid="reset-member-budget"
                onClick={() => setMemberToResetBudget(record)}
              >
                {t("teams.memberTab.columns.memberBudget.useTeamDefault")}
              </Button>
            )}
          </span>
        );
      },
    },
    {
      title: t("teams.memberTab.columns.budgetPeriod.title"),
      key: "budget_reset",
      sortValue: (record: Member) => getUserBudgetReset(record.user_id),
      render: (record: Member) => renderBudgetPeriod(record.user_id),
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("teams.memberTab.columns.rateLimits.title")}
          <SimpleTooltip content={t("teams.memberTab.columns.rateLimits.tooltip")}>
            <CircleHelp className="size-4" aria-label={t("teams.memberTab.columns.rateLimits.ariaLabel")} />
          </SimpleTooltip>
        </span>
      ),
      key: "rate_limits",
      render: (record: Member) => <span>{getUserRateLimits(record.user_id)}</span>,
    },
  ];

  const handleResetSpend = () => {
    if (!memberToResetSpend?.user_id) return;
    resetMemberSpend(
      { teamId: teamData.team_id, userId: memberToResetSpend.user_id },
      {
        onSuccess: () => {
          toast.success(t("teams.memberTab.toasts.spendReset"));
          setMemberToResetSpend(null);
          onMemberSpendReset();
        },
        onError: (error) => toast.fromError(parseErrorMessage(error)),
      },
    );
  };

  const handleResetBudget = () => {
    if (!memberToResetBudget?.user_id) return;
    resetMemberBudget(
      { teamId: teamData.team_id, userId: memberToResetBudget.user_id },
      {
        onSuccess: () => {
          toast.success(t("teams.memberTab.toasts.budgetReset"));
          setMemberToResetBudget(null);
          onMemberBudgetReset();
        },
        onError: (error) => toast.fromError(parseErrorMessage(error)),
      },
    );
  };

  const resetSpendDisabledReason = (member: Member): string | null => {
    if (!isProxyAdmin && member.user_id === userId) return t("teams.memberTab.resetSpendDisabled.ownSpend");
    if (getUserCurrentPeriodSpend(member.user_id) <= 0) return t("teams.memberTab.resetSpendDisabled.noSpend");
    return null;
  };

  const resetSpendPeriod = getUserBudgetSchedule(memberToResetSpend?.user_id ?? null).period;

  return (
    <>
      <MemberTable
        key={teamData.team_id}
        members={teamData.team_info.members_with_roles}
        canEdit={canEditTeam}
        onEdit={(record) => {
          const membership = findMembership(record.user_id);
          setSelectedEditMember(
            seedMemberBudgetFields(
              record,
              membership?.litellm_budget_table,
              teamData.team_info.team_member_budget_table?.model_max_budget,
            ),
          );
          setIsEditMemberModalVisible(true);
        }}
        onDelete={handleMemberDelete}
        onAddMember={() => setIsAddMemberModalVisible(true)}
        roleColumnTitle={t("teams.memberTab.roleColumn.title")}
        roleTooltip={t("teams.memberTab.roleColumn.tooltip")}
        extraColumns={extraColumns}
        showDeleteForMember={() =>
          isProxyAdmin || (canEditTeam && !isUserTeamAdmin) || (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)
        }
        onResetSpend={setMemberToResetSpend}
        resetSpendDisabledReason={resetSpendDisabledReason}
      />
      <Dialog open={memberToResetSpend !== null} onOpenChange={(open) => !open && setMemberToResetSpend(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("teams.memberTab.resetSpendDialog.title")}</DialogTitle>
          </DialogHeader>
          <p>
            <Trans
              i18nKey="teams.memberTab.resetSpendDialog.prompt"
              values={{ member: memberToResetSpend?.user_email || memberToResetSpend?.user_id }}
              components={{ strong: <strong /> }}
              tOptions={ESCAPED_VALUES}
              shouldUnescape
            />
          </p>
          <p className="text-sm text-muted-foreground">
            <Trans
              i18nKey={
                resetSpendPeriod
                  ? "teams.memberTab.resetSpendDialog.summaryWithPeriod"
                  : "teams.memberTab.resetSpendDialog.summary"
              }
              values={{
                period: resetSpendPeriod,
                amount: formatNumberWithCommas(getUserCurrentPeriodSpend(memberToResetSpend?.user_id ?? null), 4),
              }}
              components={{ strong: <strong /> }}
              tOptions={ESCAPED_VALUES}
              shouldUnescape
            />
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMemberToResetSpend(null)}>
              {t("teams.memberTab.resetSpendDialog.cancel")}
            </Button>
            <Button variant="destructive" onClick={handleResetSpend} disabled={isResettingSpend}>
              {t("teams.memberTab.resetSpendDialog.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={memberToResetBudget !== null} onOpenChange={(open) => !open && setMemberToResetBudget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("teams.memberTab.resetBudgetDialog.title")}</DialogTitle>
          </DialogHeader>
          <p>
            <Trans
              i18nKey="teams.memberTab.resetBudgetDialog.prompt"
              values={{
                member: memberToResetBudget?.user_email || memberToResetBudget?.user_id,
                budget: formatBudget(t, teamDefaultBudget),
              }}
              components={{ strong: <strong /> }}
              tOptions={ESCAPED_VALUES}
              shouldUnescape
            />
          </p>
          <p className="text-sm text-muted-foreground">
            <Trans
              i18nKey="teams.memberTab.resetBudgetDialog.summary"
              values={{ budget: formatBudget(t, getUserBudget(memberToResetBudget?.user_id ?? null)) }}
              components={{ strong: <strong /> }}
              tOptions={ESCAPED_VALUES}
              shouldUnescape
            />
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMemberToResetBudget(null)}>
              {t("teams.memberTab.resetBudgetDialog.cancel")}
            </Button>
            <Button onClick={handleResetBudget} disabled={isResettingBudget}>
              {t("teams.memberTab.resetBudgetDialog.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
