import React, { useState } from "react";
import {
  Modal,
  Typography,
  Divider,
  message,
  Table,
  Select,
  InputNumber,
  Card,
  Space,
  Checkbox,
} from "antd";
import { userBulkUpdateUserCall, teamBulkMemberAddCall, Member } from "./networking";
import { UserEditView } from "./user_edit_view";
import NotificationsManager from "./molecules/notifications_manager";

const { Text, Title } = Typography;

interface BulkEditUserModalProps {
  open: boolean;
  onCancel: () => void;
  selectedUsers: any[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  accessToken: string | null;
  onSuccess: () => void;
  teams: any[] | null;
  userRole: string | null;
  userModels: string[];
  allowAllUsers?: boolean; // Optional flag to enable "all users" mode
}

const BulkEditUserModal: React.FC<BulkEditUserModalProps> = ({
  open,
  onCancel,
  selectedUsers,
  possibleUIRoles,
  accessToken,
  onSuccess,
  teams,
  userRole,
  userModels,
  allowAllUsers = false,
}) => {
  const [loading, setLoading] = useState(false);
  const [selectedTeams, setSelectedTeams] = useState<string[]>([]);
  const [teamBudget, setTeamBudget] = useState<number | null>(null);
  const [addToTeams, setAddToTeams] = useState(false);
  const [updateAllUsers, setUpdateAllUsers] = useState(false);

  const handleCancel = () => {
    // Reset team management state
    setSelectedTeams([]);
    setTeamBudget(null);
    setAddToTeams(false);
    setUpdateAllUsers(false);
    onCancel();
  };

  // Create a mock userData object for the UserEditView
  const mockUserData = React.useMemo(
    () => ({
      user_id: "bulk_edit",
      user_info: {
        user_email: "",
        user_role: "",
        teams: [],
        models: [],
        max_budget: null,
        spend: 0,
        metadata: {},
        created_at: null,
        updated_at: null,
      },
      keys: [],
      teams: teams || [],
    }),
    [teams, open],
  );

  const handleSubmit = async (formValues: any) => {
    console.log("formValues", formValues);
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token not found");
      return;
    }

    setLoading(true);
    try {
      const userIds = selectedUsers.map((user) => user.user_id);

      // Build the update payload - only include fields that have been changed from default/empty values
      const updatePayload: any = {};

      if (formValues.user_role && formValues.user_role !== "") {
        updatePayload.user_role = formValues.user_role;
      }

      if (formValues.max_budget !== null && formValues.max_budget !== undefined) {
        updatePayload.max_budget = formValues.max_budget;
      }

      if (formValues.models && formValues.models.length > 0) {
        updatePayload.models = formValues.models;
      }

      if (formValues.budget_duration && formValues.budget_duration !== "") {
        updatePayload.budget_duration = formValues.budget_duration;
      }

      if (formValues.metadata && Object.keys(formValues.metadata).length > 0) {
        updatePayload.metadata = formValues.metadata;
      }

      // Check if any operations were requested
      const hasUserUpdates = Object.keys(updatePayload).length > 0;
      const hasTeamAdditions = addToTeams && selectedTeams.length > 0;

      if (!hasUserUpdates && !hasTeamAdditions) {
        NotificationsManager.fromBackend("Please modify at least one field or select teams to add users to");
        return;
      }

      let successMessages: string[] = [];

      // Handle user property updates
      if (hasUserUpdates) {
        if (updateAllUsers) {
          const result = await userBulkUpdateUserCall(accessToken, updatePayload, undefined, true);
          successMessages.push(`Updated all users (${result.total_requested} total)`);
        } else {
          await userBulkUpdateUserCall(accessToken, updatePayload, userIds);
          successMessages.push(`Updated ${userIds.length} user(s)`);
        }
      }

      // Handle team additions
      if (hasTeamAdditions) {
        const teamResults: any[] = [];

        for (const teamId of selectedTeams) {
          try {
            // Create member objects for bulk add
            let members: Member[] | null = null;
            if (updateAllUsers) {
              members = null;
            } else {
              members = selectedUsers.map((user) => ({
                user_id: user.user_id,
                role: "user" as const, // Default role for bulk add
                user_email: user.user_email || null,
              }));
            }

            const result = await teamBulkMemberAddCall(
              accessToken,
              teamId,
              members ? members : null,
              teamBudget || undefined,
              updateAllUsers,
            );

            console.log("result", result);

            teamResults.push({
              teamId,
              success: true,
              successfulAdditions: result.successful_additions,
              failedAdditions: result.failed_additions,
            });
          } catch (error) {
            console.error(`Failed to add users to team ${teamId}:`, error);
            teamResults.push({
              teamId,
              success: false,
              error: error,
            });
          }
        }

        // Generate team success message
        const successfulTeams = teamResults.filter((r) => r.success);
        const failedTeams = teamResults.filter((r) => !r.success);

        if (successfulTeams.length > 0) {
          const totalAdditions = successfulTeams.reduce((sum, r) => sum + r.successfulAdditions, 0);
          successMessages.push(`Added users to ${successfulTeams.length} team(s) (${totalAdditions} total additions)`);
        }

        if (failedTeams.length > 0) {
          message.warning(`Failed to add users to ${failedTeams.length} team(s)`);
        }
      }

      if (successMessages.length > 0) {
        NotificationsManager.success(successMessages.join(". "));
      }

      // Reset team management state
      setSelectedTeams([]);
      setTeamBudget(null);
      setAddToTeams(false);
      setUpdateAllUsers(false);

      onSuccess();
      onCancel();
    } catch (error) {
      console.error("Bulk operation failed:", error);
      NotificationsManager.fromBackend("Failed to perform bulk operations");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      open={open}
      onCancel={handleCancel}
      footer={null}
      title={updateAllUsers ? "Bulk Edit All Users" : `Bulk Edit ${selectedUsers.length} User(s)`}
      width={800}
    >
      {allowAllUsers && (
        <div className="mb-4">
          <Checkbox checked={updateAllUsers} onChange={(e) => setUpdateAllUsers(e.target.checked)}>
            <Text strong>Update ALL users in the system</Text>
          </Checkbox>
          {updateAllUsers && (
            <div style={{ marginTop: 8 }}>
              <Text type="warning" style={{ fontSize: "12px" }}>
                ⚠️ This will apply changes to ALL users in the system, not just the selected ones.
              </Text>
            </div>
          )}
        </div>
      )}

      {!updateAllUsers && (
        <div className="mb-4">
          <Title level={5}>Selected Users ({selectedUsers.length}):</Title>
          <Table
            size="small"
            bordered
            dataSource={selectedUsers}
            pagination={false}
            scroll={{ y: 200 }}
            rowKey="user_id"
            columns={[
              {
                title: "User ID",
                dataIndex: "user_id",
                key: "user_id",
                width: "30%",
                render: (text: string) => (
                  <Text strong style={{ fontSize: "12px" }}>
                    {text.length > 20 ? `${text.slice(0, 20)}...` : text}
                  </Text>
                ),
              },
              {
                title: "Email",
                dataIndex: "user_email",
                key: "user_email",
                width: "25%",
                render: (text: string) => (
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    {text || "No email"}
                  </Text>
                ),
              },
              {
                title: "Current Role",
                dataIndex: "user_role",
                key: "user_role",
                width: "25%",
                render: (role: string) => (
                  <Text style={{ fontSize: "12px" }}>{possibleUIRoles?.[role]?.ui_label || role}</Text>
                ),
              },
              {
                title: "Budget",
                dataIndex: "max_budget",
                key: "max_budget",
                width: "20%",
                render: (budget: number | null) => (
                  <Text style={{ fontSize: "12px" }}>{budget !== null ? `$${budget}` : "Unlimited"}</Text>
                ),
              },
            ]}
          />
        </div>
      )}

      <Divider />

      <div className="mb-4">
        <Text>
          <strong>Instructions:</strong> Fill in the fields below with the values you want to apply to all selected
          users. You can bulk edit: role, budget, models, and metadata. You can also add users to teams.
        </Text>
      </div>

      {/* Team Management Section */}
      <Card title="Team Management" size="small" className="mb-4" style={{ backgroundColor: "#fafafa" }}>
        <Space direction="vertical" style={{ width: "100%" }}>
          <Checkbox checked={addToTeams} onChange={(e) => setAddToTeams(e.target.checked)}>
            Add selected users to teams
          </Checkbox>

          {addToTeams && (
            <>
              <div>
                <Text strong>Select Teams:</Text>
                <Select
                  mode="multiple"
                  placeholder="Select teams to add users to"
                  value={selectedTeams}
                  onChange={setSelectedTeams}
                  style={{ width: "100%", marginTop: 8 }}
                  options={
                    teams?.map((team) => ({
                      label: team.team_alias || team.team_id,
                      value: team.team_id,
                    })) || []
                  }
                />
              </div>

              <div>
                <Text strong>Team Budget (Optional):</Text>
                <InputNumber
                  placeholder="Max budget per user in team"
                  value={teamBudget}
                  onChange={(value) => setTeamBudget(value)}
                  style={{ width: "100%", marginTop: 8 }}
                  min={0}
                  step={0.01}
                  precision={2}
                />
                <Text type="secondary" style={{ fontSize: "12px" }}>
                  Leave empty for unlimited budget within team limits
                </Text>
              </div>

              <Text type="secondary" style={{ fontSize: "12px" }}>
                Users will be added with &quot;user&quot; role by default. All users will be added to each selected
                team.
              </Text>
            </>
          )}
        </Space>
      </Card>

      <UserEditView
        userData={mockUserData}
        onCancel={handleCancel}
        onSubmit={handleSubmit}
        teams={teams}
        accessToken={accessToken}
        userID="bulk_edit"
        userRole={userRole}
        userModels={userModels}
        possibleUIRoles={possibleUIRoles}
        isBulkEdit={true}
      />

      {loading && (
        <div style={{ textAlign: "center", marginTop: "10px" }}>
          <Text>Updating {updateAllUsers ? "all users" : selectedUsers.length} user(s)...</Text>
        </div>
      )}
    </Modal>
  );
};

export default BulkEditUserModal;
