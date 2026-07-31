import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Member } from "@/components/networking";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import NotificationsManager from "@/components/molecules/notifications_manager";
import NumericalInput from "@/components/shared/numerical_input";
import { DateCell, MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, Checkbox, Form, Modal, Select, Space, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import { useState } from "react";
import MemberTable from "@/components/common_components/MemberTable";
import { TeamMemberBulkUpdateFields, teamMemberBulkUpdateCall } from "./teamMemberBulkUpdate";
import { TeamData } from "./TeamInfo";

interface TeamMemberTabProps {
  teamData: TeamData;
  canEditTeam: boolean;
  handleMemberDelete: (member: Member) => void;
  setSelectedEditMember: (member: Member) => void;
  setIsEditMemberModalVisible: (visible: boolean) => void;
  setIsAddMemberModalVisible: (visible: boolean) => void;
  onMembersUpdated?: () => Promise<void>;
}

export default function TeamMemberTab({
  teamData,
  canEditTeam,
  handleMemberDelete,
  setSelectedEditMember,
  setIsEditMemberModalVisible,
  setIsAddMemberModalVisible,
  onMembersUpdated,
}: TeamMemberTabProps) {
  const [isBulkUpdateVisible, setIsBulkUpdateVisible] = useState(false);
  const [isBulkUpdating, setIsBulkUpdating] = useState(false);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedMembers, setSelectedMembers] = useState<Member[]>([]);
  const [bulkUpdateForm] = Form.useForm();
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

    const rpmText = rpmLimit ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit ? `${formatNumber(tpmLimit)} TPM` : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : "No Limits";
  };

  const { data: uiSettingsData } = useUISettings();
  const { accessToken, userId, userRole } = useAuthorized();
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
              <Typography.Text key={m} code style={{ fontSize: "12px" }}>
                {m}
              </Typography.Text>
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
        <MoneyCell value={getUserCurrentCycleSpend(record.user_id)} decimals={4} />
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
      render: (_: unknown, record: Member) => <MoneyCell value={getUserTotalSpend(record.user_id)} decimals={4} />,
    },
    {
      title: "Team Member Budget (USD)",
      key: "budget",
      render: (_: unknown, record: Member) => (
        <MoneyCell value={getUserBudget(record.user_id)} decimals={4} emptyText="Unlimited" showZero />
      ),
    },
    {
      title: "Budget Reset",
      key: "budget_reset",
      render: (_: unknown, record: Member) => <DateCell value={getUserBudgetReset(record.user_id)} precision="date" />,
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
      render: (_: unknown, record: Member) => <Typography.Text>{getUserRateLimits(record.user_id)}</Typography.Text>,
    },
  ];

  const handleBulkUpdate = async (values: {
    apply_role?: boolean;
    role?: "admin" | "user";
    apply_max_budget?: boolean;
    max_budget_in_team?: number | null;
    apply_budget_duration?: boolean;
    budget_duration?: string | null;
    apply_tpm_limit?: boolean;
    tpm_limit?: number | null;
    apply_rpm_limit?: boolean;
    rpm_limit?: number | null;
    apply_allowed_models?: boolean;
    allowed_models?: string[];
  }) => {
    if (!accessToken) return;
    const userIds = selectedMembers.flatMap((member) => (member.user_id ? [member.user_id] : []));
    if (userIds.length === 0) {
      NotificationsManager.fromBackend("Select at least one team member");
      return;
    }
    const updateFields: TeamMemberBulkUpdateFields = {
      ...(values.apply_role ? { role: values.role } : {}),
      ...(values.apply_max_budget ? { max_budget_in_team: values.max_budget_in_team ?? null } : {}),
      ...(values.apply_budget_duration ? { budget_duration: values.budget_duration ?? null } : {}),
      ...(values.apply_tpm_limit ? { tpm_limit: values.tpm_limit ?? null } : {}),
      ...(values.apply_rpm_limit ? { rpm_limit: values.rpm_limit ?? null } : {}),
      ...(values.apply_allowed_models ? { allowed_models: values.allowed_models ?? [] } : {}),
    };
    if (Object.keys(updateFields).length === 0) {
      NotificationsManager.fromBackend("Select at least one field to update");
      return;
    }

    setIsBulkUpdating(true);
    try {
      const response = await teamMemberBulkUpdateCall(accessToken, teamData.team_id, userIds, updateFields);
      await onMembersUpdated?.();
      setIsBulkUpdateVisible(false);
      setSelectedMembers([]);
      setSelectionMode(false);
      bulkUpdateForm.resetFields();
      NotificationsManager.success(
        `${response.successful_updates.length} team member${response.successful_updates.length === 1 ? "" : "s"} updated`,
      );
      if (response.failed_updates.length > 0) {
        NotificationsManager.fromBackend(`${response.failed_updates.length} team member updates failed`);
      }
    } catch (error) {
      NotificationsManager.fromBackend(error instanceof Error ? error.message : "Failed to bulk update team members");
    } finally {
      setIsBulkUpdating(false);
    }
  };

  return (
    <>
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
        extraActions={
          <>
            <Button
              onClick={() => {
                setSelectionMode((current) => !current);
                setSelectedMembers([]);
              }}
              type={selectionMode ? "primary" : "default"}
            >
              {selectionMode ? "Cancel Selection" : "Select Members"}
            </Button>
            {selectionMode && (
              <Button
                type="primary"
                disabled={selectedMembers.length === 0}
                onClick={() => setIsBulkUpdateVisible(true)}
              >
                Bulk Edit ({selectedMembers.length} selected)
              </Button>
            )}
          </>
        }
        roleColumnTitle="Team Role"
        roleTooltip="This role applies only to this team and is independent from the user's proxy-level role."
        extraColumns={extraColumns}
        rowSelection={
          selectionMode
            ? {
                selectedRowKeys: selectedMembers.flatMap((member) => (member.user_id ? [member.user_id] : [])),
                onChange: (_selectedRowKeys, selectedRows) => setSelectedMembers(selectedRows),
                getCheckboxProps: (member) => ({ disabled: member.user_id === null }),
              }
            : undefined
        }
        showDeleteForMember={() =>
          isProxyAdmin || (canEditTeam && !isUserTeamAdmin) || (isUserTeamAdmin && !disableTeamAdminDeleteTeamUser)
        }
      />
      <Modal
        title={`Bulk Edit ${selectedMembers.length} Team Member${selectedMembers.length === 1 ? "" : "s"}`}
        open={isBulkUpdateVisible}
        onCancel={() => setIsBulkUpdateVisible(false)}
        onOk={() => bulkUpdateForm.submit()}
        okText="Update Members"
        confirmLoading={isBulkUpdating}
      >
        <Form form={bulkUpdateForm} layout="vertical" onFinish={handleBulkUpdate}>
          <Typography.Text type="secondary">Choose the fields to apply to every selected member.</Typography.Text>
          <Form.Item name="apply_role" valuePropName="checked" className="mt-3 mb-1">
            <Checkbox>Team role</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_role") && (
                <Form.Item name="role" rules={[{ required: true, message: "Select a role" }]}>
                  <Select
                    placeholder="Select role"
                    options={[
                      { label: "Admin", value: "admin" },
                      { label: "User", value: "user" },
                    ]}
                  />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item name="apply_max_budget" valuePropName="checked" className="mb-1">
            <Checkbox>Team member budget (USD)</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_max_budget") && (
                <Form.Item name="max_budget_in_team">
                  <NumericalInput
                    min={0}
                    step={0.01}
                    placeholder="Leave blank to clear the budget"
                    style={{ width: "100%" }}
                  />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item name="apply_budget_duration" valuePropName="checked" className="mb-1">
            <Checkbox>Budget reset period</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_budget_duration") && (
                <Form.Item name="budget_duration">
                  <BudgetDurationDropdown />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item name="apply_tpm_limit" valuePropName="checked" className="mb-1">
            <Checkbox>TPM limit</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_tpm_limit") && (
                <Form.Item name="tpm_limit">
                  <NumericalInput min={0} placeholder="Leave blank to clear the TPM limit" style={{ width: "100%" }} />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item name="apply_rpm_limit" valuePropName="checked" className="mb-1">
            <Checkbox>RPM limit</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_rpm_limit") && (
                <Form.Item name="rpm_limit">
                  <NumericalInput min={0} placeholder="Leave blank to clear the RPM limit" style={{ width: "100%" }} />
                </Form.Item>
              )
            }
          </Form.Item>
          <Form.Item name="apply_allowed_models" valuePropName="checked" className="mb-1">
            <Checkbox>Allowed models</Checkbox>
          </Form.Item>
          <Form.Item noStyle shouldUpdate>
            {({ getFieldValue }) =>
              getFieldValue("apply_allowed_models") && (
                <Form.Item name="allowed_models">
                  <Select
                    mode="multiple"
                    placeholder="Leave empty to inherit team models"
                    options={(teamData.team_info.models || []).map((model) => ({ label: model, value: model }))}
                  />
                </Form.Item>
              )
            }
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}
