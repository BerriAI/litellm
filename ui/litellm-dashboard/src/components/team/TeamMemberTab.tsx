import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import { DateCell, MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Space, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import MemberTable from "@/components/common_components/MemberTable";
import { TeamData } from "./TeamInfo";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation("gateway");
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
    if (!userId) return t("teams.memberTable.noLimits");
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit ? `${formatNumber(tpmLimit)} TPM` : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : t("teams.memberTable.noLimits");
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

  const extraColumns: ColumnsType<Member> = [
    {
      title: (
        <Space direction="horizontal">
          {t("teams.memberTable.modelScope")}
          <Tooltip title={t("teams.memberTable.modelScopeTooltip")}>
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "model_scope",
      render: (_: unknown, record: Member) => {
        const models = getUserAllowedModels(record.user_id);
        if (!models) {
          return <Typography.Text type="secondary">({t("teams.myMembership.allTeamModels")})</Typography.Text>;
        }
        const displayed = models.slice(0, 2);
        const remaining = models.length - displayed.length;
        return (
          <Space wrap>
            {displayed.map((m) => (
              <Typography.Text key={m} code style={{ fontSize: "12px" }}>
                {m}
              </Typography.Text>
            ))}
            {remaining > 0 && (
              <Tooltip title={models.slice(2).join(", ")}>
                <Typography.Text type="secondary">{t("teams.available.more", { count: remaining })}</Typography.Text>
              </Tooltip>
            )}
          </Space>
        );
      },
    },
    {
      title: (
        <Space direction="horizontal">
          {t("teams.myMembership.currentSpend")}
          <Tooltip title={t("teams.memberTable.currentSpendTooltip")}>
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "spend",
      render: (_: unknown, record: Member) => (
        <MoneyCell value={getUserCurrentCycleSpend(record.user_id)} decimals={4} />
      ),
    },
    {
      title: (
        <Space direction="horizontal">
          {t("teams.myMembership.totalSpend")}
          <Tooltip title={t("teams.memberTable.totalSpendTooltip")}>
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "total_spend",
      render: (_: unknown, record: Member) => <MoneyCell value={getUserTotalSpend(record.user_id)} decimals={4} />,
    },
    {
      title: t("teams.create.memberBudget"),
      key: "budget",
      render: (_: unknown, record: Member) => (
        <MoneyCell value={getUserBudget(record.user_id)} decimals={4} emptyText={t("teams.table.unlimited")} showZero />
      ),
    },
    {
      title: t("teams.details.settings.budgetReset"),
      key: "budget_reset",
      render: (_: unknown, record: Member) => <DateCell value={getUserBudgetReset(record.user_id)} precision="date" />,
    },
    {
      title: (
        <Space direction="horizontal">
          {t("teams.memberTable.rateLimits")}
          <Tooltip title={t("teams.memberTable.rateLimitsTooltip")}>
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      key: "rate_limits",
      render: (_: unknown, record: Member) => <Typography.Text>{getUserRateLimits(record.user_id)}</Typography.Text>,
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
          max_budget_in_team: membership?.litellm_budget_table?.max_budget || null,
          tpm_limit: membership?.litellm_budget_table?.tpm_limit || null,
          rpm_limit: membership?.litellm_budget_table?.rpm_limit || null,
          budget_duration: membership?.litellm_budget_table?.budget_duration || null,
          allowed_models: membership?.litellm_budget_table?.allowed_models || [],
        };
        setSelectedEditMember(enhancedMember);
        setIsEditMemberModalVisible(true);
      }}
      onDelete={handleMemberDelete}
      onAddMember={() => setIsAddMemberModalVisible(true)}
      roleColumnTitle={t("teams.myMembership.teamRole")}
      roleTooltip={t("teams.memberTable.roleTooltip")}
      extraColumns={extraColumns}
      showDeleteForMember={() =>
        isProxyAdmin || (canEditTeam && !isUserTeamAdmin) || (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)
      }
    />
  );
}
