import { useLanguage } from "@/contexts/LanguageContext";
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
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { MoneyCell } from "@/components/shared/table_cells";
import { toast } from "@/lib/toast";
import { describeBudgetResetSchedule } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { CircleHelp } from "lucide-react";
import { useMemo, useState, type ComponentProps } from "react";
import { TeamData, TeamMemberBudgetSource, TeamMembership } from "./TeamInfo";

const BUDGET_SOURCE_LABELS: Record<Exclude<TeamMemberBudgetSource, "none">, string> = {
  team_default: "Team default",
  custom: "Custom",
};

const formatBudget = (value: number | null): string =>
  value === null ? "Unlimited" : `$${formatNumberWithCommas(value, 2)}`;

// 确认提示中的成员标识属于业务内容，按邮箱、ID 顺序原样显示。
const memberLabel = (member: Member | null): string => member?.user_email || member?.user_id || "";

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
  const { t } = useLanguage();
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

  const getUserCurrentCycleSpend = (userId: string | null): number => {
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
    if (!userId) return t("No Limits");
    const membership = findMembership(userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit != null ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit != null ? `${formatNumber(tpmLimit)} TPM` : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : t("No Limits");
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

  const renderBudgetReset = (userId: string | null) => {
    const budget = findMembership(userId)?.litellm_budget_table;
    const schedule = describeBudgetResetSchedule(budget?.budget_duration ?? null, budget?.budget_reset_at ?? null, t);
    return (
      <SimpleTooltip content={schedule.tooltip}>
        <span>{schedule.label}</span>
      </SimpleTooltip>
    );
  };

  const extraColumns: NonNullable<ComponentProps<typeof MemberTable>["extraColumns"]> = [
    {
      title: (
        <span className="flex items-center gap-1">
          {t("Model Scope")}
          <SimpleTooltip content={t("Models this member can access. Empty means they inherit all team models.")}>
            <CircleHelp className="size-4" aria-label={t("Model scope information")} />
          </SimpleTooltip>
        </span>
      ),
      key: "model_scope",
      render: (record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return <span className="text-muted-foreground">{t("(all team models)")}</span>;
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
                  +{remaining} {t("more")}
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
          {t("Current Cycle Spend (USD)")}
          <SimpleTooltip
            content={t(
              "Spend in this member's current budget cycle, the value checked against their budget. The Budget Reset column shows when this cycle ends, or Never resets. Spend above a rollover cap at reset carries into the next cycle.",
            )}
          >
            <CircleHelp className="size-4" aria-label={t("Current cycle spend information")} />
          </SimpleTooltip>
        </span>
      ),
      key: "spend",
      numeric: true,
      sortValue: (record: Member) => getUserCurrentCycleSpend(record.user_id),
      render: (record: Member) => <MoneyCell value={getUserCurrentCycleSpend(record.user_id)} decimals={2} />,
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("Total Spend (USD)")}
          <SimpleTooltip
            content={t(
              "Cumulative spend by this member within this team, across budget cycles. This is a lower bound: lifetime tracking began 2026-04-21, so cycles completed before then are not included. It is never less than Current Cycle Spend.",
            )}
          >
            <CircleHelp className="size-4" aria-label={t("Total spend information")} />
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
          {t("Team Member Budget (USD)")}
          <SimpleTooltip
            content={t(
              "Team default follows the team's member budget, so changing it in team settings updates this member too. Custom is set on this member only and ignores later team changes.",
            )}
          >
            <CircleHelp className="size-4" aria-label={t("Team member budget information")} />
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
            <MoneyCell value={getUserBudget(record.user_id)} decimals={2} emptyText={t("Unlimited")} showZero />
            {source !== "none" && (
              <Badge variant={source === "custom" ? "outline" : "secondary"} data-testid="member-budget-source">
                {t(BUDGET_SOURCE_LABELS[source])}
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
                {t("Use team default")}
              </Button>
            )}
          </span>
        );
      },
    },
    {
      title: t("Budget Reset"),
      key: "budget_reset",
      sortValue: (record: Member) => getUserBudgetReset(record.user_id),
      render: (record: Member) => renderBudgetReset(record.user_id),
    },
    {
      title: (
        <span className="flex items-center gap-1">
          {t("Team Member Rate Limits")}
          <SimpleTooltip content={t("Rate limits for this member's usage within this team.")}>
            <CircleHelp className="size-4" aria-label={t("Team member rate limits information")} />
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
          toast.success(<TranslatedMessage message="Team member spend reset to $0" />);
          setMemberToResetSpend(null);
          onMemberSpendReset();
        },
        // 原始错误保留状态分类，只有缺少消息时使用随语言切换的本地提示。
        onError: (error: unknown) => {
          const hasEmptyMessage = error instanceof Error && !error.message;
          toast.fromError(
            error,
            error == null || error === "" || hasEmptyMessage
              ? { description: <TranslatedMessage message="Failed to reset team member spend" /> }
              : undefined,
          );
        },
      },
    );
  };

  const handleResetBudget = () => {
    if (!memberToResetBudget?.user_id) return;
    resetMemberBudget(
      { teamId: teamData.team_id, userId: memberToResetBudget.user_id },
      {
        onSuccess: () => {
          toast.success(<TranslatedMessage message="Team member budget reset to the team default" />);
          setMemberToResetBudget(null);
          onMemberBudgetReset();
        },
        onError: (error: unknown) => {
          const hasEmptyMessage = error instanceof Error && !error.message;
          toast.fromError(
            error,
            error == null || error === "" || hasEmptyMessage
              ? { description: <TranslatedMessage message="Failed to reset team member budget" /> }
              : undefined,
          );
        },
      },
    );
  };

  const resetSpendDisabledReason = (member: Member): string | null => {
    if (!isProxyAdmin && member.user_id === userId) return t("Ask a proxy admin to reset your own spend");
    if (getUserCurrentCycleSpend(member.user_id) <= 0) return t("No current cycle spend to reset");
    return null;
  };

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
        roleColumnTitle={t("Team Role")}
        roleTooltip={t("This role applies only to this team and is independent from the user's proxy-level role.")}
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
            <DialogTitle>{t("Reset Team Member Spend")}</DialogTitle>
          </DialogHeader>
          <p>
            {t("Reset the current cycle spend for {member} in this team to $0?", {
              member: memberLabel(memberToResetSpend),
            })}
          </p>
          <p className="text-sm text-muted-foreground">
            {t("Current cycle spend:")}{" "}
            <strong>${formatNumberWithCommas(getUserCurrentCycleSpend(memberToResetSpend?.user_id ?? null), 4)}</strong>
            {t(". This is the value checked against the member's budget. Total spend and logs are preserved.")}
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMemberToResetSpend(null)}>
              {t("Cancel")}
            </Button>
            <Button variant="destructive" onClick={handleResetSpend} disabled={isResettingSpend}>
              {t("Reset")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <Dialog open={memberToResetBudget !== null} onOpenChange={(open) => !open && setMemberToResetBudget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("Reset Team Member Budget")}</DialogTitle>
          </DialogHeader>
          <p>
            {t("Remove the custom budget for {member} and use the team default of {budget}?", {
              member: memberLabel(memberToResetBudget),
              budget: teamDefaultBudget === null ? t("Unlimited") : formatBudget(teamDefaultBudget),
            })}
          </p>
          <p className="text-sm text-muted-foreground">
            {t("Custom budget:")}{" "}
            <strong>
              {getUserBudget(memberToResetBudget?.user_id ?? null) === null
                ? t("Unlimited")
                : formatBudget(getUserBudget(memberToResetBudget?.user_id ?? null))}
            </strong>
            {t(". Their spend is kept. Future changes to the team's member budget will apply to them again.")}
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setMemberToResetBudget(null)}>
              {t("Cancel")}
            </Button>
            <Button onClick={handleResetBudget} disabled={isResettingBudget}>
              {t("Use team default")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
