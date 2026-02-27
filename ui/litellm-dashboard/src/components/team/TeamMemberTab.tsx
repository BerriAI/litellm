import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Space, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import MemberTable from "@/components/common_components/MemberTable";
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

  const extraColumns: ColumnsType<Member> = [
    {
      title: (
        <Space direction="horizontal">
          Team Member Spend (USD)
          <Tooltip title="This is the amount spent by a user in the team.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "spend",
      render: (_: unknown, record: Member) => (
        <Typography.Text>${formatNumberWithCommas(getUserSpend(record.user_id), 4)}</Typography.Text>
      ),
    },
    {
      title: "Team Member Budget (USD)",
      key: "budget",
      render: (_: unknown, record: Member) => {
        const budget = getUserBudget(record.user_id);
        return (
          <Typography.Text>
            {budget ? `$${formatNumberWithCommas(Number(budget), 4)}` : "No Limit"}
          </Typography.Text>
        );
      },
    },
    {
      title: (
        <Space direction="horizontal">
          Team Member Rate Limits
          <Tooltip title="Rate limits for this member's usage within this team.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "rate_limits",
      render: (_: unknown, record: Member) => (
        <Typography.Text>{getUserRateLimits(record.user_id)}</Typography.Text>
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
        isProxyAdmin || (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)
      }
    />
  );
}
