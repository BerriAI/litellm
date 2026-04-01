import React, { useState } from "react";
import {
  Card, Text, Button, Grid, Tab, TabList, TabGroup, TabPanel, TabPanels, Title,
  Table, TableHead, TableBody, TableRow, TableHeaderCell, TableCell,
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon, RefreshIcon, PlusIcon } from "@heroicons/react/outline";
import {
  userGetInfoV2,
  UserInfoV2Response,
  userDeleteCall,
  userUpdateUserCall,
  modelAvailableCall,
  invitationCreateCall,
  getProxyBaseUrl,
  teamInfoCall,
  teamListCall,
  teamMemberAddCall,
  teamMemberDeleteCall,
  Member,
} from "../networking";
import { Button as AntdButton, Modal, Select as AntdSelect, Form, Tooltip } from "antd";
import { rolesWithWriteAccess } from "../../utils/roles";
import { UserEditView } from "../user_edit_view";
import OnboardingModal, { InvitationLink } from "../onboarding_link";
import { formatNumberWithCommas, copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CopyIcon, CheckIcon } from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown";
import DeleteResourceModal from "../common_components/DeleteResourceModal";

interface UserInfoViewProps {
  userId: string;
  onClose: () => void;
  accessToken: string | null;
  userRole: string | null;
  onDelete?: () => void;
  possibleUIRoles: Record<string, Record<string, string>> | null;
  initialTab?: number; // 0 for Overview, 1 for Details
  startInEditMode?: boolean;
}

/** Team info used for display in user detail view */
interface TeamDisplayInfo {
  team_id: string;
  team_alias: string | null;
}

export default function UserInfoView({
  userId,
  onClose,
  accessToken,
  userRole,
  onDelete,
  possibleUIRoles,
  initialTab = 0,
  startInEditMode = false,
}: UserInfoViewProps) {
  const [userData, setUserData] = useState<UserInfoV2Response | null>(null);
  const [teamDetails, setTeamDetails] = useState<TeamDisplayInfo[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeletingUser, setIsDeletingUser] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(startInEditMode);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(initialTab);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isTeamsExpanded, setIsTeamsExpanded] = useState(false);
  const [isAddTeamModalOpen, setIsAddTeamModalOpen] = useState(false);
  const [isRemoveTeamModalOpen, setIsRemoveTeamModalOpen] = useState(false);
  const [teamToRemove, setTeamToRemove] = useState<TeamDisplayInfo | null>(null);
  const [isAddingTeam, setIsAddingTeam] = useState(false);
  const [isRemovingTeam, setIsRemovingTeam] = useState(false);
  const [allTeams, setAllTeams] = useState<Array<{ team_id: string; team_alias: string }>>([]);
  const [selectedTeamId, setSelectedTeamId] = useState<string>("");
  const [selectedRole, setSelectedRole] = useState<string>("user");
  const [isLoadingTeams, setIsLoadingTeams] = useState(false);

  React.useEffect(() => {
    setBaseUrl(getProxyBaseUrl());
  }, []);

  React.useEffect(() => {
    console.log(`userId: ${userId}, userRole: ${userRole}, accessToken: ${accessToken}`);
    const fetchData = async () => {
      try {
        if (!accessToken) return;
        const data = await userGetInfoV2(accessToken, userId);
        setUserData(data);

        // Fetch team details for display (team aliases)
        if (data.teams && data.teams.length > 0) {
          try {
            const teamPromises = data.teams.map(async (teamId: string) => {
              try {
                const teamData = await teamInfoCall(accessToken, teamId);
                return {
                  team_id: teamId,
                  team_alias: teamData?.team_info?.team_alias || null,
                };
              } catch {
                return { team_id: teamId, team_alias: null };
              }
            });
            const teams = await Promise.all(teamPromises);
            setTeamDetails(teams);
          } catch {
            // Fall back to just team IDs
            setTeamDetails(data.teams.map((id: string) => ({ team_id: id, team_alias: null })));
          }
        }

        // Fetch available models
        const modelDataResponse = await modelAvailableCall(accessToken, userId, userRole || "");
        const availableModels = modelDataResponse.data.map((model: any) => model.id);
        setUserModels(availableModels);
      } catch (error) {
        console.error("Error fetching user data:", error);
        NotificationsManager.fromBackend("Failed to fetch user data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [accessToken, userId, userRole]);

  const isProxyAdmin = userRole === "proxy_admin" || userRole === "Admin";

  const fetchAllTeams = async () => {
    if (!accessToken) return;
    setIsLoadingTeams(true);
    try {
      const teams = await teamListCall(accessToken, null);
      setAllTeams(
        (teams || []).map((t: any) => ({
          team_id: t.team_id,
          team_alias: t.team_alias || t.team_id,
        }))
      );
    } catch (error) {
      console.error("Error fetching teams:", error);
    } finally {
      setIsLoadingTeams(false);
    }
  };

  const handleOpenAddTeamModal = () => {
    setSelectedTeamId("");
    setSelectedRole("user");
    setIsAddTeamModalOpen(true);
    fetchAllTeams();
  };

  const handleAddTeamSubmit = async () => {
    if (!accessToken || !selectedTeamId) return;
    setIsAddingTeam(true);
    try {
      const member: Member = {
        role: selectedRole,
        user_id: userId,
      };
      await teamMemberAddCall(accessToken, selectedTeamId, member);
      NotificationsManager.success("User added to team successfully");
      setIsAddTeamModalOpen(false);
      // Re-fetch user data to refresh teams
      const data = await userGetInfoV2(accessToken, userId);
      setUserData(data);
      if (data.teams && data.teams.length > 0) {
        const teamPromises = data.teams.map(async (teamId: string) => {
          try {
            const teamData = await teamInfoCall(accessToken, teamId);
            return { team_id: teamId, team_alias: teamData?.team_info?.team_alias || null };
          } catch {
            return { team_id: teamId, team_alias: null };
          }
        });
        setTeamDetails(await Promise.all(teamPromises));
      } else {
        setTeamDetails([]);
      }
    } catch (error: any) {
      console.error("Error adding user to team:", error);
      NotificationsManager.fromBackend(error?.message || "Failed to add user to team");
    } finally {
      setIsAddingTeam(false);
    }
  };

  const handleOpenRemoveTeamModal = (team: TeamDisplayInfo) => {
    setTeamToRemove(team);
    setIsRemoveTeamModalOpen(true);
  };

  const handleRemoveTeamConfirm = async () => {
    if (!accessToken || !teamToRemove) return;
    setIsRemovingTeam(true);
    try {
      const member: Member = {
        role: "user",
        user_id: userId,
      };
      await teamMemberDeleteCall(accessToken, teamToRemove.team_id, member);
      NotificationsManager.success("User removed from team successfully");
      setIsRemoveTeamModalOpen(false);
      setTeamToRemove(null);
      // Re-fetch user data to refresh teams
      const data = await userGetInfoV2(accessToken, userId);
      setUserData(data);
      if (data.teams && data.teams.length > 0) {
        const teamPromises = data.teams.map(async (teamId: string) => {
          try {
            const teamData = await teamInfoCall(accessToken, teamId);
            return { team_id: teamId, team_alias: teamData?.team_info?.team_alias || null };
          } catch {
            return { team_id: teamId, team_alias: null };
          }
        });
        setTeamDetails(await Promise.all(teamPromises));
      } else {
        setTeamDetails([]);
      }
    } catch (error: any) {
      console.error("Error removing user from team:", error);
      NotificationsManager.fromBackend(error?.message || "Failed to remove user from team");
    } finally {
      setIsRemovingTeam(false);
    }
  };

  const handleRemoveTeamCancel = () => {
    setIsRemoveTeamModalOpen(false);
    setTeamToRemove(null);
  };

  const availableTeamsForAdd = allTeams.filter(
    (t) => !teamDetails.some((td) => td.team_id === t.team_id)
  );

  const handleResetPassword = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token not found");
      return;
    }
    try {
      NotificationsManager.success("Generating password reset link...");
      const data = await invitationCreateCall(accessToken, userId);
      setInvitationLinkData(data);
      setIsInvitationLinkModalVisible(true);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to generate password reset link");
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      setIsDeletingUser(true);
      await userDeleteCall(accessToken, [userId]);
      NotificationsManager.success("User deleted successfully");
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting user:", error);
      NotificationsManager.fromBackend("Failed to delete user");
    } finally {
      setIsDeleteModalOpen(false);
      setIsDeletingUser(false);
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
  };

  const handleUserUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken || !userData) return;

      const response = await userUpdateUserCall(accessToken, formValues, null);

      // Update local state with new values
      setUserData({
        ...userData,
        user_email: formValues.user_email ?? userData.user_email,
        user_alias: formValues.user_alias ?? userData.user_alias,
        models: formValues.models ?? userData.models,
        max_budget: formValues.max_budget ?? userData.max_budget,
        budget_duration: formValues.budget_duration ?? userData.budget_duration,
        metadata: formValues.metadata ?? userData.metadata,
      });

      NotificationsManager.success("User updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating user:", error);
      NotificationsManager.fromBackend("Failed to update user");
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Users
        </Button>
        <Text>Loading user data...</Text>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="p-4">
        <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Users
        </Button>
        <Text>User not found</Text>
      </div>
    );
  }

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  // Build a legacy-compatible shape for UserEditView
  const userDataForEdit = {
    user_id: userData.user_id,
    user_info: {
      user_email: userData.user_email,
      user_alias: userData.user_alias,
      user_role: userData.user_role,
      models: userData.models,
      max_budget: userData.max_budget,
      budget_duration: userData.budget_duration,
      metadata: userData.metadata,
    },
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
            Back to Users
          </Button>
          <Title>{userData.user_email || "User"}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{userData.user_id}</Text>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["user-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(userData.user_id, "user-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["user-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <div className="flex items-center space-x-2">
            <Button icon={RefreshIcon} variant="secondary" onClick={handleResetPassword} className="flex items-center">
              Reset Password
            </Button>
            <Button
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center text-red-500 border-red-500 hover:text-red-600 hover:border-red-600"
            >
              Delete User
            </Button>
          </div>
        )}
      </div>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete User?"
        message="Are you sure you want to delete this user? This action cannot be undone."
        resourceInformationTitle="User Information"
        resourceInformation={[
          { label: "Email", value: userData.user_email },
          { label: "User ID", value: userData.user_id, code: true },
          {
            label: "Global Proxy Role",
            value:
              (userData.user_role && possibleUIRoles?.[userData.user_role]?.ui_label) ||
              userData.user_role ||
              "-",
          },
          {
            label: "Total Spend (USD)",
            value:
              userData.spend !== null && userData.spend !== undefined
                ? userData.spend.toFixed(2)
                : undefined,
          },
        ]}
        onCancel={cancelDelete}
        onOk={handleDelete}
        confirmLoading={isDeletingUser}
      />

      <TabGroup defaultIndex={activeTab} onIndexChange={setActiveTab}>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Details</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Spend</Text>
                <div className="mt-2">
                  <Title>${formatNumberWithCommas(userData.spend || 0, 4)}</Title>
                  <Text>
                    of{" "}
                    {userData.max_budget !== null
                      ? `$${formatNumberWithCommas(userData.max_budget, 4)}`
                      : "Unlimited"}
                  </Text>
                </div>
              </Card>

              <Card>
                <div className="flex justify-between items-center mb-2">
                  <Text>Teams</Text>
                  {isProxyAdmin && (
                    <Button
                      icon={PlusIcon}
                      variant="light"
                      size="xs"
                      onClick={handleOpenAddTeamModal}
                    >
                      Add Team
                    </Button>
                  )}
                </div>
                <div className="mt-2">
                  {teamDetails.length > 0 ? (
                    <div className="max-h-60 overflow-y-auto">
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableHeaderCell>Team Name</TableHeaderCell>
                          {isProxyAdmin && <TableHeaderCell className="text-right">Actions</TableHeaderCell>}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {teamDetails.slice(0, isTeamsExpanded ? teamDetails.length : 20).map((team) => (
                          <TableRow key={team.team_id}>
                            <TableCell>{team.team_alias || team.team_id}</TableCell>
                            {isProxyAdmin && (
                              <TableCell className="text-right">
                                <Button
                                  icon={TrashIcon}
                                  variant="light"
                                  size="xs"
                                  color="red"
                                  onClick={() => handleOpenRemoveTeamModal(team)}
                                />
                              </TableCell>
                            )}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    </div>
                  ) : (
                    <Text>No teams</Text>
                  )}
                  {!isTeamsExpanded && teamDetails.length > 20 && (
                    <Button
                      variant="light"
                      size="xs"
                      className="mt-2"
                      onClick={() => setIsTeamsExpanded(true)}
                    >
                      +{teamDetails.length - 20} more
                    </Button>
                  )}
                  {isTeamsExpanded && teamDetails.length > 20 && (
                    <Button
                      variant="light"
                      size="xs"
                      className="mt-2"
                      onClick={() => setIsTeamsExpanded(false)}
                    >
                      Show Less
                    </Button>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Personal Models</Text>
                <div className="mt-2">
                  {userData.models?.length && userData.models?.length > 0 ? (
                    userData.models?.map((model, index) => <Text key={index}>{model}</Text>)
                  ) : (
                    <Text>All proxy models</Text>
                  )}
                </div>
              </Card>
            </Grid>
          </TabPanel>

          {/* Details Panel */}
          <TabPanel>
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>User Settings</Title>
                {!isEditing && userRole && rolesWithWriteAccess.includes(userRole) && (
                  <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
                )}
              </div>

              {isEditing && userData ? (
                <UserEditView
                  userData={userDataForEdit}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleUserUpdate}
                  teams={teamDetails}
                  accessToken={accessToken}
                  userID={userId}
                  userRole={userRole}
                  userModels={userModels}
                  possibleUIRoles={possibleUIRoles}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">User ID</Text>
                    <div className="flex items-center cursor-pointer">
                      <Text className="font-mono">{userData.user_id}</Text>
                      <AntdButton
                        type="text"
                        size="small"
                        icon={copiedStates["user-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
                        onClick={() => copyToClipboard(userData.user_id, "user-id")}
                        className={`left-2 z-10 transition-all duration-200 ${
                          copiedStates["user-id"]
                            ? "text-green-600 bg-green-50 border-green-200"
                            : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
                        }`}
                      />
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Email</Text>
                    <Text>{userData.user_email || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">User Alias</Text>
                    <Text>{userData.user_alias || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Global Proxy Role</Text>
                    <Text>{userData.user_role || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Created</Text>
                    <Text>
                      {userData.created_at
                        ? new Date(userData.created_at).toLocaleString()
                        : "Unknown"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Last Updated</Text>
                    <Text>
                      {userData.updated_at
                        ? new Date(userData.updated_at).toLocaleString()
                        : "Unknown"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Personal Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.models?.length && userData.models?.length > 0 ? (
                        userData.models?.map((model, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-100 rounded text-xs">
                            {model}
                          </span>
                        ))
                      ) : (
                        <Text>All proxy models</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Max Budget</Text>
                    <Text>
                      {userData.max_budget !== null && userData.max_budget !== undefined
                        ? `$${formatNumberWithCommas(userData.max_budget, 4)}`
                        : "Unlimited"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Budget Reset</Text>
                    <Text>{getBudgetDurationLabel(userData.budget_duration ?? null)}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Metadata</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {JSON.stringify(userData.metadata || {}, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
      <OnboardingModal
        isInvitationLinkModalVisible={isInvitationLinkModalVisible}
        setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
        baseUrl={baseUrl || ""}
        invitationLinkData={invitationLinkData}
        modalType="resetPassword"
      />

      {/* Delete Team Member Modal */}
      <DeleteResourceModal
        isOpen={isRemoveTeamModalOpen}
        title="Remove from Team"
        alertMessage="Removing this user from the team will also delete any keys the user created for this team."
        message="Are you sure you want to remove this user from the team? This action cannot be undone."
        resourceInformationTitle="Team Membership"
        resourceInformation={[
          { label: "Team", value: teamToRemove?.team_alias || teamToRemove?.team_id },
          { label: "User ID", value: userData?.user_id, code: true },
          { label: "Email", value: userData?.user_email },
        ]}
        onCancel={handleRemoveTeamCancel}
        onOk={handleRemoveTeamConfirm}
        confirmLoading={isRemovingTeam}
      />

      {/* Add to Team Modal */}
      <Modal
        title="Add User to Team"
        open={isAddTeamModalOpen}
        onCancel={() => setIsAddTeamModalOpen(false)}
        footer={null}
        width={500}
        maskClosable={!isAddingTeam}
      >
        <Form
          layout="vertical"
          onFinish={handleAddTeamSubmit}
        >
          <Form.Item label="Team" required>
            <AntdSelect
              showSearch
              value={selectedTeamId || undefined}
              onChange={setSelectedTeamId}
              placeholder="Select a team"
              filterOption={(input, option) => {
                const team = availableTeamsForAdd.find((t) => t.team_id === option?.value);
                if (!team) return false;
                return team.team_alias.toLowerCase().includes(input.toLowerCase());
              }}
              loading={isLoadingTeams}
            >
              {availableTeamsForAdd.map((team) => (
                <AntdSelect.Option key={team.team_id} value={team.team_id}>
                  {team.team_alias}
                </AntdSelect.Option>
              ))}
            </AntdSelect>
          </Form.Item>

          <Form.Item label="Member Role">
            <AntdSelect value={selectedRole} onChange={setSelectedRole}>
              <AntdSelect.Option value="user">
                <Tooltip title="Can view team info, but not manage it">
                  <span className="font-medium">user</span>
                  <span className="ml-2 text-gray-500 text-sm">- Can view team info, but not manage it</span>
                </Tooltip>
              </AntdSelect.Option>
              <AntdSelect.Option value="admin">
                <Tooltip title="Can create team keys, add members, and manage settings">
                  <span className="font-medium">admin</span>
                  <span className="ml-2 text-gray-500 text-sm">- Can create team keys, add members, and manage settings</span>
                </Tooltip>
              </AntdSelect.Option>
            </AntdSelect>
          </Form.Item>

          <div className="text-right mt-4">
            <AntdButton
              type="primary"
              htmlType="submit"
              loading={isAddingTeam}
              disabled={!selectedTeamId}
            >
              {isAddingTeam ? "Adding..." : "Add to Team"}
            </AntdButton>
          </div>
        </Form>
      </Modal>
    </div>
  );
}
