import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import { formatBudgetReset } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Input, Select, Space, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import MemberTable from "@/components/common_components/MemberTable";
import { TeamData, TeamMembership } from "./TeamInfo";
import { useEffect, useMemo, useState } from "react";

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
  const [searchText, setSearchText] = useState("");
  const [roleFilter, setRoleFilter] = useState<string | null>(null);
  const [memberTablePage, setMemberTablePage] = useState(1);

  // Reset to page 1 when filter/search changes — without remounting MemberTable
  useEffect(() => { setMemberTablePage(1); }, [searchText, roleFilter]);

  // O(1) lookup instead of O(n) find() per member per column
  const membershipsMap = useMemo(
    () =>
      new Map<string, TeamMembership>(
        teamData.team_memberships
          .filter((tm) => tm.user_id)
          .map((tm) => [tm.user_id, tm]),
      ),
    [teamData.team_memberships],
  );

  const filteredMembers = useMemo(() => {
    const q = searchText.trim().toLowerCase();
    return teamData.team_info.members_with_roles.filter((m) => {
      if (roleFilter) {
        const role = m.role?.toLowerCase() ?? "";
        const isAdmin = role === "admin" || role === "org_admin";
        if (roleFilter === "admin" && !isAdmin) return false;
        if (roleFilter === "non-admin" && isAdmin) return false;
      }
      if (!q) return true;
      return (
        m.user_email?.toLowerCase().includes(q) ||
        m.user_id?.toLowerCase().includes(q)
      );
    });
  }, [teamData.team_info.members_with_roles, searchText, roleFilter]);

  const formatNumber = (value: number | null): string => {
    if (value === null || value === undefined) return "0";
    if (typeof value === "number") {
      const normalNumber = Number(value);
      if (normalNumber === Math.floor(normalNumber)) return normalNumber.toString();
      return formatNumberWithCommas(normalNumber, 8).replace(/\.?0+$/, "");
    }
    return "0";
  };

  const getUserCurrentCycleSpend = (userId: string | null): number => {
    if (!userId) return 0;
    return membershipsMap.get(userId)?.spend ?? 0;
  };

  const getUserTotalSpend = (userId: string | null): number => {
    if (!userId) return 0;
    return membershipsMap.get(userId)?.total_spend ?? 0;
  };

  const getUserBudget = (userId: string | null): string | null => {
    if (!userId) return null;
    const maxBudget = membershipsMap.get(userId)?.litellm_budget_table?.max_budget;
    if (maxBudget === null || maxBudget === undefined) return null;
    return formatNumber(maxBudget);
  };

  const getUserRateLimits = (userId: string | null): string => {
    if (!userId) return "No Limits";
    const membership = membershipsMap.get(userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;
    const rpmText = rpmLimit ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit ? `${formatNumber(tpmLimit)} TPM` : null;
    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : "No Limits";
  };

  const getUserAllowedModels = (userId: string | null): string[] | null => {
    if (!userId) return null;
    const models = membershipsMap.get(userId)?.litellm_budget_table?.allowed_models;
    return models && models.length > 0 ? models : null;
  };

  const getUserBudgetReset = (userId: string | null): string | null => {
    if (!userId) return null;
    return formatBudgetReset(membershipsMap.get(userId)?.litellm_budget_table?.budget_reset_at);
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
          Model Scope
          <Tooltip title="Models this member can access. Empty means they inherit all team models.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "model_scope",
      render: (_: unknown, record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return <Typography.Text type="secondary">(all team models)</Typography.Text>;
        }
        const displayed = models.slice(0, 2);
        const remaining = models.length - displayed.length;
        return (
          <Space wrap>
            {displayed.map((m) => (
              <Typography.Text key={m} code style={{ fontSize: "12px" }}>{m}</Typography.Text>
            ))}
            {remaining > 0 && (
              <Tooltip title={models.slice(2).join(", ")}>
                <Typography.Text type="secondary">+{remaining} more</Typography.Text>
              </Tooltip>
            )}
          </Space>
        );
      },
    },
    {
      title: (
        <Space direction="horizontal">
          Current Cycle Spend (USD)
          <Tooltip title="Spend for the current budget cycle. Resets to $0 when the member's budget window rolls over. This is the value checked against the member's budget.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "spend",
      render: (_: unknown, record: Member) => (
        <Typography.Text>${formatNumberWithCommas(getUserCurrentCycleSpend(record.user_id), 4)}</Typography.Text>
      ),
    },
    {
      title: (
        <Space direction="horizontal">
          Total Spend (USD)
          <Tooltip title="Cumulative spend by this member within this team, across all budget cycles. Tracking began 2026-04-21; spend from before that date is not included.">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "total_spend",
      render: (_: unknown, record: Member) => (
        <Typography.Text>${formatNumberWithCommas(getUserTotalSpend(record.user_id), 4)}</Typography.Text>
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
      title: "Budget Reset",
      key: "budget_reset",
      render: (_: unknown, record: Member) => {
        const reset = getUserBudgetReset(record.user_id);
        return reset ? (
          <Typography.Text>{reset}</Typography.Text>
        ) : (
          <Typography.Text type="secondary">—</Typography.Text>
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
    <Space direction="vertical" style={{ width: "100%" }}>
      <Space wrap>
        <Input.Search
          placeholder="Search by email or user ID"
          allowClear
          style={{ width: 280 }}
          onChange={(e) => setSearchText(e.target.value)}
          onSearch={(v) => setSearchText(v)}
        />
        <Select
          placeholder="Filter by role"
          allowClear
          style={{ width: 160 }}
          options={[
            { value: "admin", label: "Admin" },
            { value: "non-admin", label: "Non-admin" },
          ]}
          onChange={(v) => setRoleFilter(v ?? null)}
        />
      </Space>
      <MemberTable
        members={filteredMembers}
        canEdit={canEditTeam}
        withPagination
        currentPage={memberTablePage}
        onPageChange={setMemberTablePage}
        onEdit={(record) => {
          const membership = membershipsMap.get(record.user_id ?? "");
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
    </Space>
  );
}
