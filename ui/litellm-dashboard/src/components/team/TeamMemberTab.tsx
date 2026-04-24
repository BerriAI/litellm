import React from "react";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import MemberTable, {
  MemberTableExtraColumn,
} from "@/components/common_components/MemberTable";
import { TeamData } from "./TeamInfo";

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

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

  // Helper function to get spend for a user
  const getUserSpend = (userId: string | null): number | null => {
    if (!userId) return 0;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.spend || 0;
  };

  const getUserBudget = (userId: string | null): string | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const maxBudget = membership?.litellm_budget_table?.max_budget;
    if (maxBudget === null || maxBudget === undefined) {
      return null;
    }
    return formatNumber(maxBudget);
  };

  // Helper function to get rate limits for a user
  const getUserRateLimits = (userId: string | null): string => {
    if (!userId) return "No Limits";
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit ? `${formatNumber(tpmLimit)} TPM` : null;

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

  const extraColumns: MemberTableExtraColumn[] = [
    {
      title: (
        <span className="inline-flex items-center gap-2">
          Model Scope
          <InfoTip>
            Models this member can access. Empty means they inherit all team
            models.
          </InfoTip>
        </span>
      ),
      key: "model_scope",
      render: (_: unknown, record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return (
            <span className="text-muted-foreground">(all team models)</span>
          );
        }
        const displayed = models.slice(0, 2);
        const remaining = models.length - displayed.length;
        return (
          <div className="flex flex-wrap gap-2 items-center">
            {displayed.map((m) => (
              <code key={m} className="text-xs">
                {m}
              </code>
            ))}
            {remaining > 0 && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="text-muted-foreground">
                      +{remaining} more
                    </span>
                  </TooltipTrigger>
                  <TooltipContent>{models.slice(2).join(", ")}</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        );
      },
    },
    {
      title: (
        <span className="inline-flex items-center gap-2">
          Team Member Spend (USD)
          <InfoTip>This is the amount spent by a user in the team.</InfoTip>
        </span>
      ),
      key: "spend",
      render: (_: unknown, record: Member) => (
        <span>${formatNumberWithCommas(getUserSpend(record.user_id), 4)}</span>
      ),
    },
    {
      title: "Team Member Budget (USD)",
      key: "budget",
      render: (_: unknown, record: Member) => {
        const budget = getUserBudget(record.user_id);
        return (
          <span>
            {budget ? `$${formatNumberWithCommas(Number(budget), 4)}` : "No Limit"}
          </span>
        );
      },
    },
    {
      title: (
        <span className="inline-flex items-center gap-2">
          Team Member Rate Limits
          <InfoTip>Rate limits for this member&apos;s usage within this team.</InfoTip>
        </span>
      ),
      key: "rate_limits",
      render: (_: unknown, record: Member) => (
        <span>{getUserRateLimits(record.user_id)}</span>
      ),
    },
  ];

  return (
    <MemberTable
      members={teamData.team_info.members_with_roles}
      canEdit={canEditTeam}
      onEdit={(record) => {
        const membership = teamData.team_memberships.find(
          (tm) => tm.user_id === record.user_id
        );
        const enhancedMember = {
          ...record,
          max_budget_in_team: membership?.litellm_budget_table?.max_budget || null,
          tpm_limit: membership?.litellm_budget_table?.tpm_limit || null,
          rpm_limit: membership?.litellm_budget_table?.rpm_limit || null,
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
