import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { CrownOutlined, InfoCircleOutlined, UserAddOutlined, UserOutlined } from "@ant-design/icons";
import { Button, Space, Table, Tag, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { TeamData } from "./TeamInfo";

const { Text } = Typography;

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

  const columns: ColumnsType<Member> = [
    {
      title: "User Email",
      dataIndex: "user_email",
      key: "user_email",
      render: (email: string | null) => (
        <Text>{email || "-"}</Text>
      ),
    },
    {
      title: "User ID",
      dataIndex: "user_id",
      key: "user_id",
      render: (userId: string | null) =>
        userId === "default_user_id" ? (
          <Tag color="blue">Default Proxy Admin</Tag>
        ) : (
          <Text>{userId}</Text>
        ),
    },
    {
      title: (
        <Space direction="horizontal">
          Team Role
          <Tooltip title="This role applies only to this team and is independent from the user's proxy-level role.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: "role",
      key: "role",
      render: (role: string) => (
        <Space>
          {role?.toLowerCase() === "admin" ? (
            <CrownOutlined />
          ) : (
            <UserOutlined />
          )}
          <Text style={{ textTransform: "capitalize" }}>{role}</Text>
        </Space>
      ),
    },
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
        <Text>
          ${formatNumberWithCommas(getUserSpend(record.user_id), 4)}
        </Text>
      ),
    },
    {
      title: "Team Member Budget (USD)",
      key: "budget",
      render: (_: unknown, record: Member) => {
        const budget = getUserBudget(record.user_id);
        return (
          <Text >
            {budget ? `$${formatNumberWithCommas(Number(budget), 4)}` : "No Limit"}
          </Text>
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
        <Text>{getUserRateLimits(record.user_id)}</Text>
      ),
    },
    {
      title: "Actions",
      key: "actions",
      fixed: "right",
      width: 120,
      render: (_: unknown, record: Member) =>
        canEditTeam ? (
          <div className="flex gap-2">
            <TableIconActionButton
              variant="Edit"
              tooltipText="Edit member"
              dataTestId="edit-member"
              onClick={() => {
                const membership = teamData.team_memberships.find(
                  (tm) => tm.user_id === record.user_id
                );
                const enhancedMember = {
                  ...record,
                  max_budget_in_team:
                    membership?.litellm_budget_table?.max_budget || null,
                  tpm_limit:
                    membership?.litellm_budget_table?.tpm_limit || null,
                  rpm_limit:
                    membership?.litellm_budget_table?.rpm_limit || null,
                };
                setSelectedEditMember(enhancedMember);
                setIsEditMemberModalVisible(true);
              }}
            />
            {(isProxyAdmin ||
              (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)) && (
                <TableIconActionButton
                  variant="Delete"
                  tooltipText="Delete member"
                  dataTestId="delete-member"
                  onClick={() => handleMemberDelete(record)}
                />
              )}
          </div>
        ) : null,
    },
  ];

  return (
    <div className="space-y-4">
      <Table
        columns={columns}
        dataSource={teamData.team_info.members_with_roles}
        rowKey={(record, index) => record.user_id || String(index)}
        pagination={false}
        size="small"
        scroll={{ x: "max-content" }}
      />
      <Button
        icon={<UserAddOutlined />}
        type="primary"
        onClick={() => setIsAddMemberModalVisible(true)}
      >
        Add Member
      </Button>
    </div>
  );
};
