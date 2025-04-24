import React, { useState } from "react";
import {
  Card,
  Text,
  Button,
  Grid,
  Col,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Title,
  Badge,
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon } from "@heroicons/react/outline";
import { userInfoCall, userDeleteCall, userUpdateUserCall, modelAvailableCall } from "../networking";
import { message } from "antd";
import { rolesWithWriteAccess } from '../../utils/roles';
import { UserEditView } from "../user_edit_view";

interface UserInfoViewProps {
  userId: string;
  onClose: () => void;
  accessToken: string | null;
  userRole: string | null;
  onDelete?: () => void;
  possibleUIRoles: Record<string, Record<string, string>> | null;
}

interface UserInfo {
  user_id: string;
  user_info: {
    user_email: string | null;
    user_role: string | null;
    teams: any[] | null;
    models: string[] | null;
    max_budget: number | null;
    spend: number | null;
    metadata: Record<string, any> | null;
    created_at: string | null;
    updated_at: string | null;
  };
  keys: any[] | null;
  teams: any[] | null;
}

export default function UserInfoView({ userId, onClose, accessToken, userRole, onDelete, possibleUIRoles }: UserInfoViewProps) {
  const [userData, setUserData] = useState<UserInfo | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);

  React.useEffect(() => {
    console.log(`userId: ${userId}, userRole: ${userRole}, accessToken: ${accessToken}`)
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
        message.error("Failed to fetch user data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [accessToken, userId, userRole]);

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      await userDeleteCall(accessToken, [userId]);
      message.success("User deleted successfully");
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting user:", error);
      message.error("Failed to delete user");
    }
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
          models: formValues.models,
          max_budget: formValues.max_budget,
          metadata: formValues.metadata,
        }
      });

      message.success("User updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating user:", error);
      message.error("Failed to update user");
    }
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
          variant="light"
          onClick={onClose}
          className="mb-4"
        >
          Back to Users
        </Button>
        <Text>Loading user data...</Text>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
          variant="light"
          onClick={onClose}
          className="mb-4"
        >
          Back to Users
        </Button>
        <Text>User not found</Text>
      </div>
    );
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button 
            icon={ArrowLeftIcon} 
            variant="light"
            onClick={onClose}
            className="mb-4"
          >
            Back to Users
          </Button>
          <Title>{userData.user_info?.user_email || "User"}</Title>
          <Text className="text-gray-500 font-mono">{userData.user_id}</Text>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <Button
            icon={TrashIcon}
            variant="secondary"
            onClick={() => setIsDeleteModalOpen(true)}
            className="flex items-center"
          >
            Delete User
          </Button>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">
                      Delete User
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this user?
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button
                  onClick={handleDelete}
                  color="red"
                  className="ml-2"
                >
                  Delete
                </Button>
                <Button onClick={() => setIsDeleteModalOpen(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      <TabGroup>
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
                  <Title>${Number(userData.user_info?.spend || 0).toFixed(4)}</Title>
                  <Text>of {userData.user_info?.max_budget !== null ? `$${userData.user_info.max_budget}` : "Unlimited"}</Text>
                </div>
              </Card>

              <Card>
                <Text>Teams</Text>
                <div className="mt-2">
                  <Text>{userData.teams?.length || 0} teams</Text>
                </div>
              </Card>

              <Card>
                <Text>API Keys</Text>
                <div className="mt-2">
                  <Text>{userData.keys?.length || 0} keys</Text>
                </div>
              </Card>

              <Card>
                <Text>Personal Models</Text>
                <div className="mt-2">
                  {userData.user_info?.models?.length && userData.user_info?.models?.length > 0 ? (
                    userData.user_info?.models?.map((model, index) => (
                      <Text key={index}>{model}</Text>
                    ))
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
                  <Button variant="light" onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
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
                    <Text className="font-mono">{userData.user_id}</Text>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Email</Text>
                    <Text>{userData.user_info?.user_email || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Role</Text>
                    <Text>{userData.user_info?.user_role || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Created</Text>
                    <Text>{userData.user_info?.created_at ? new Date(userData.user_info.created_at).toLocaleString() : "Unknown"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Last Updated</Text>
                    <Text>{userData.user_info?.updated_at ? new Date(userData.user_info.updated_at).toLocaleString() : "Unknown"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Teams</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.teams?.length && userData.teams?.length > 0 ? (
                        userData.teams?.map((team, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-blue-100 rounded text-xs"
                          >
                            {team.team_alias || team.team_id}
                          </span>
                        ))
                      ) : (
                        <Text>No teams</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.user_info?.models?.length && userData.user_info?.models?.length > 0 ? (
                        userData.user_info?.models?.map((model, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-blue-100 rounded text-xs"
                          >
                            {model}
                          </span>
                        ))
                      ) : (
                        <Text>All proxy models</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">API Keys</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.keys?.length && userData.keys?.length > 0 ? (
                        userData.keys.map((key, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-green-100 rounded text-xs"
                          >
                            {key.key_alias || key.token}
                          </span>
                        ))
                      ) : (
                        <Text>No API keys</Text>
                      )}
                    </div>
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
    </div>
  );
} 