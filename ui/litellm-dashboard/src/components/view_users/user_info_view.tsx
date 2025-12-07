import React, { useState } from "react";
import { Card, Text, Button, Grid, Tab, TabList, TabGroup, TabPanel, TabPanels, Title, Badge } from "@tremor/react";
import { ArrowLeftIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import {
  userInfoCall,
  userDeleteCall,
  userUpdateUserCall,
  modelAvailableCall,
  invitationCreateCall,
  getProxyBaseUrl,
} from "../networking";
import { Button as AntdButton } from "antd";
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

interface UserInfo {
  user_id: string;
  user_info: {
    user_email: string | null;
    user_alias: string | null;
    user_role: string | null;
    teams: any[] | null;
    models: string[] | null;
    max_budget: number | null;
    budget_duration: string | null;
    spend: number | null;
    metadata: Record<string, any> | null;
    created_at: string | null;
    updated_at: string | null;
  };
  keys: any[] | null;
  teams: any[] | null;
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
  const [userData, setUserData] = useState<UserInfo | null>(null);
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

  React.useEffect(() => {
    setBaseUrl(getProxyBaseUrl());
  }, []);

  React.useEffect(() => {
    console.log(`userId: ${userId}, userRole: ${userRole}, accessToken: ${accessToken}`);
    const fetchData = async () => {
      try {
        if (!accessToken) return;
        const data = await userInfoCall(accessToken, userId, userRole || "", false, null, null, true);
        setUserData(data);

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
        user_info: {
          ...userData.user_info,
          user_email: formValues.user_email,
          user_alias: formValues.user_alias,
          models: formValues.models,
          max_budget: formValues.max_budget,
          budget_duration: formValues.budget_duration,
          metadata: formValues.metadata,
        },
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

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
            Back to Users
          </Button>
          <Title>{userData.user_info?.user_email || "User"}</Title>
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
          { label: "Email", value: userData.user_info?.user_email },
          { label: "User ID", value: userData.user_id, code: true },
          {
            label: "Global Proxy Role",
            value:
              (userData.user_info?.user_role && possibleUIRoles?.[userData.user_info.user_role]?.ui_label) ||
              userData.user_info?.user_role ||
              "-",
          },
          {
            label: "Total Spend (USD)",
            value:
              userData.user_info?.spend !== null && userData.user_info?.spend !== undefined
                ? userData.user_info.spend.toFixed(2)
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
                  <Title>${formatNumberWithCommas(userData.user_info?.spend || 0, 4)}</Title>
                  <Text>
                    of{" "}
                    {userData.user_info?.max_budget !== null
                      ? `$${formatNumberWithCommas(userData.user_info.max_budget, 4)}`
                      : "Unlimited"}
                  </Text>
                </div>
              </Card>

              <Card>
                <Text>Teams</Text>
                <div className="mt-2">
                  {userData.teams?.length && userData.teams?.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {userData.teams?.slice(0, isTeamsExpanded ? userData.teams.length : 20).map((team, index) => (
                        <Badge key={index} color="blue" title={team.team_alias}>
                          {team.team_alias}
                        </Badge>
                      ))}
                      {!isTeamsExpanded && userData.teams?.length > 20 && (
                        <Badge
                          color="gray"
                          className="cursor-pointer hover:bg-gray-200 transition-colors"
                          onClick={() => setIsTeamsExpanded(true)}
                        >
                          +{userData.teams.length - 20} more
                        </Badge>
                      )}
                      {isTeamsExpanded && userData.teams?.length > 20 && (
                        <Badge
                          color="gray"
                          className="cursor-pointer hover:bg-gray-200 transition-colors"
                          onClick={() => setIsTeamsExpanded(false)}
                        >
                          Show Less
                        </Badge>
                      )}
                    </div>
                  ) : (
                    <Text>No teams</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Virtual Keys</Text>
                <div className="mt-2">
                  <Text>
                    {userData.keys?.length || 0} {userData.keys?.length === 1 ? "Key" : "Keys"}
                  </Text>
                </div>
              </Card>

              <Card>
                <Text>Personal Models</Text>
                <div className="mt-2">
                  {userData.user_info?.models?.length && userData.user_info?.models?.length > 0 ? (
                    userData.user_info?.models?.map((model, index) => <Text key={index}>{model}</Text>)
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
                  userData={userData}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleUserUpdate}
                  teams={userData.teams}
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
                    <Text>{userData.user_info?.user_email || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">User Alias</Text>
                    <Text>{userData.user_info?.user_alias || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Global Proxy Role</Text>
                    <Text>{userData.user_info?.user_role || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Created</Text>
                    <Text>
                      {userData.user_info?.created_at
                        ? new Date(userData.user_info.created_at).toLocaleString()
                        : "Unknown"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Last Updated</Text>
                    <Text>
                      {userData.user_info?.updated_at
                        ? new Date(userData.user_info.updated_at).toLocaleString()
                        : "Unknown"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Teams</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.teams?.length && userData.teams?.length > 0 ? (
                        <>
                          {userData.teams?.slice(0, isTeamsExpanded ? userData.teams.length : 20).map((team, index) => (
                            <span
                              key={index}
                              className="px-2 py-1 bg-blue-100 rounded text-xs"
                              title={team.team_alias || team.team_id}
                            >
                              {team.team_alias || team.team_id}
                            </span>
                          ))}
                          {!isTeamsExpanded && userData.teams?.length > 20 && (
                            <span
                              className="px-2 py-1 bg-gray-100 rounded text-xs cursor-pointer hover:bg-gray-200 transition-colors"
                              onClick={() => setIsTeamsExpanded(true)}
                            >
                              +{userData.teams.length - 20} more
                            </span>
                          )}
                          {isTeamsExpanded && userData.teams?.length > 20 && (
                            <span
                              className="px-2 py-1 bg-gray-100 rounded text-xs cursor-pointer hover:bg-gray-200 transition-colors"
                              onClick={() => setIsTeamsExpanded(false)}
                            >
                              Show Less
                            </span>
                          )}
                        </>
                      ) : (
                        <Text>No teams</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Personal Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.user_info?.models?.length && userData.user_info?.models?.length > 0 ? (
                        userData.user_info?.models?.map((model, index) => (
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
                    <Text className="font-medium">Virtual Keys</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.keys?.length && userData.keys?.length > 0 ? (
                        userData.keys.map((key, index) => (
                          <span key={index} className="px-2 py-1 bg-green-100 rounded text-xs">
                            {key.key_alias || key.token}
                          </span>
                        ))
                      ) : (
                        <Text>No Virtual Keys</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Max Budget</Text>
                    <Text>
                      {userData.user_info?.max_budget !== null && userData.user_info?.max_budget !== undefined
                        ? `$${formatNumberWithCommas(userData.user_info.max_budget, 4)}`
                        : "Unlimited"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Budget Reset</Text>
                    <Text>{getBudgetDurationLabel(userData.user_info?.budget_duration ?? null)}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Metadata</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {JSON.stringify(userData.user_info?.metadata || {}, null, 2)}
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
    </div>
  );
}
