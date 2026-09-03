import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { SimpleTooltip } from "@/components/ui/tooltip";
import MemberTable from "@/components/common_components/MemberTable";
import { Member } from "@/components/networking";
import { DateCell, MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { CircleHelp } from "lucide-react";
import type { ComponentProps } from "react";
import { TeamData } from "./TeamInfo";

interface TeamMemberTabProps {
  teamData: TeamData;
  canEditTeam: boolean;
  handleMemberDelete: (member: Member) => void;
  setSelectedEditMember: (member: Member) => void;
  setIsEditMemberModalVisible: (visible: boolean) => void;
  setIsAddMemberModalVisible: (visible: boolean) => void;
}

export default function TeamMemberTab({
  teamData,
  canEditTeam,
  handleMemberDelete,
  setSelectedEditMember,
  setIsEditMemberModalVisible,
  setIsAddMemberModalVisible,
}: TeamMemberTabProps) {
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
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.spend ?? 0;
  };

  const getUserTotalSpend = (userId: string | null): number => {
    if (!userId) return 0;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.total_spend ?? 0;
  };

  const getUserBudget = (userId: string | null): number | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.litellm_budget_table?.max_budget ?? null;
  };

  // Helper function to get rate limits for a user
  const getUserRateLimits = (userId: string | null): string => {
    if (!userId) return "No Limits";
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit != null ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit != null ? `${formatNumber(tpmLimit)} TPM` : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : "No Limits";
  };

  const { data: uiSettingsData } = useUISettings();
  const { userId, userRole } = useAuthorized();
  const disableTeamAdminDeleteTeamUser = Boolean(uiSettingsData?.values?.disable_team_admin_delete_team_user);
  const isUserTeamAdmin = isUserTeamAdminForSingleTeam(teamData.team_info.members_with_roles, userId || "");
  const isProxyAdmin = isProxyAdminRole(userRole || "");

  const getUserAllowedModels = (userId: string | null): string[] | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const models = membership?.litellm_budget_table?.allowed_models;
    return models && models.length > 0 ? models : null;
  };

  const getUserBudgetReset = (userId: string | null): string | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.litellm_budget_table?.budget_reset_at ?? null;
  };

  const extraColumns: NonNullable<ComponentProps<typeof MemberTable>["extraColumns"]> = [
    {
      title: (
        <span className="flex items-center gap-1">
          Model Scope
          <SimpleTooltip content="Models this member can access. Empty means they inherit all team models.">
            <CircleHelp className="size-4" aria-label="Model scope information" />
          </SimpleTooltip>
        </span>
      ),
      key: "model_scope",
      render: (_: unknown, record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return <span className="text-muted-foreground">(all team models)</span>;
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
                <span className="text-muted-foreground">+{remaining} more</span>
              </SimpleTooltip>
            )}
          </div>
        );
      },
    },
    {
      title: (
        <span className="flex items-center gap-1">
          Current Cycle Spend (USD)
          <SimpleTooltip content="Spend for the current budget cycle. Resets to $0 when the member's budget window rolls over. This is the value checked against the member's budget.">
            <CircleHelp className="size-4" aria-label="Current cycle spend information" />
          </SimpleTooltip>
        </span>
      ),
      key: "spend",
      render: (_: unknown, record: Member) => (
        <MoneyCell value={getUserCurrentCycleSpend(record.user_id)} decimals={2} />
      ),
    },
    {
      title: (
        <span className="flex items-center gap-1">
          Total Spend (USD)
          <SimpleTooltip content="Cumulative spend by this member within this team, across all budget cycles. Tracking began 2026-04-21; spend from before that date is not included.">
            <CircleHelp className="size-4" aria-label="Total spend information" />
          </SimpleTooltip>
        </span>
      ),
      key: "total_spend",
      render: (_: unknown, record: Member) => <MoneyCell value={getUserTotalSpend(record.user_id)} decimals={2} />,
    },
    {
      title: "Team Member Budget (USD)",
      key: "budget",
      render: (_: unknown, record: Member) => (
        <MoneyCell value={getUserBudget(record.user_id)} decimals={2} emptyText="Unlimited" showZero />
      ),
    },
    {
      title: "Budget Reset",
      key: "budget_reset",
      render: (_: unknown, record: Member) => <DateCell value={getUserBudgetReset(record.user_id)} precision="date" />,
    },
    {
      title: (
        <span className="flex items-center gap-1">
          Team Member Rate Limits
          <SimpleTooltip content="Rate limits for this member's usage within this team.">
            <CircleHelp className="size-4" aria-label="Team member rate limits information" />
          </SimpleTooltip>
        </span>
      ),
      key: "rate_limits",
      render: (_: unknown, record: Member) => <span>{getUserRateLimits(record.user_id)}</span>,
    },
  ];

  return (
    <MemberTable
      members={teamData.team_info.members_with_roles}
      canEdit={canEditTeam}
      onEdit={(record) => {
        const membership = teamData.team_memberships.find((tm) => tm.user_id === record.user_id);
        const enhancedMember = {
          ...record,
          max_budget_in_team: membership?.litellm_budget_table?.max_budget ?? null,
          tpm_limit: membership?.litellm_budget_table?.tpm_limit ?? null,
          rpm_limit: membership?.litellm_budget_table?.rpm_limit ?? null,
          budget_duration: membership?.litellm_budget_table?.budget_duration || null,
          allowed_models: membership?.litellm_budget_table?.allowed_models || [],
        };
        setSelectedEditMember(enhancedMember);
        setIsEditMemberModalVisible(true);
      }}
      onDelete={handleMemberDelete}
      onAddMember={() => setIsAddMemberModalVisible(true)}
      roleColumnTitle="Team Role"
      roleTooltip="This role applies only to this team and is independent from the user's proxy-level role."
      extraColumns={extraColumns}
      showDeleteForMember={() =>
        isProxyAdmin || (canEditTeam && !isUserTeamAdmin) || (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)
      }
    />
  );
}
