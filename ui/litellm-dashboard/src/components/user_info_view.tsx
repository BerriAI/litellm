import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
  Button as TremorButton,
  TextInput,
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon } from "@heroicons/react/outline";
import { userUpdateUserCall, userDeleteCall } from "./networking";
import { Button, Form, Input, message, Select } from "antd";

interface UserInfoViewProps {
  userId: string;
  onClose: () => void;
  userData: any;
  accessToken: string | null;
  userRole: string | null;
  onUserUpdated: () => Promise<void>;
}

export default function UserInfoView({ 
  userId, 
  onClose, 
  userData, 
  accessToken,
  userRole,
  onUserUpdated
}: UserInfoViewProps) {
  const [form] = Form.useForm();
  const [localUserData, setLocalUserData] = useState(userData);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  const canEditUser = userRole === "Admin" || userRole === "Proxy Admin";

  const handleUserUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      setIsSaving(true);
      
      const updateData = {
        user_id: userId,
        user_email: values.user_email,
        user_role: values.user_role,
        max_budget: values.max_budget,
        spend: values.spend,
        team_id: values.team_id,
      };

      await userUpdateUserCall(accessToken, updateData, null);
      
      setLocalUserData({
        ...localUserData,
        ...updateData
      });

      message.success("User settings updated successfully");
      setIsDirty(false);
      setIsEditing(false);
      await onUserUpdated();
    } catch (error) {
      console.error("Error updating user:", error);
      message.error("Failed to update user settings");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return;
      await userDeleteCall(accessToken, [userId]);
      message.success("User deleted successfully");
      onClose();
      await onUserUpdated();
    } catch (error) {
      console.error("Error deleting the user:", error);
      message.error("Failed to delete user");
    }
  };

  if (!userData) {
    return (
      <div className="p-4">
        <Button 
          icon={<ArrowLeftIcon />}
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
            icon={<ArrowLeftIcon />}
            onClick={onClose}
            className="mb-4"
          >
            Back to Users
          </Button>
          <Title>User: {userData.user_id}</Title>
          <Text className="text-gray-500">{userData.user_email || "No email provided"}</Text>
        </div>
        {canEditUser && (
          <div className="flex gap-2">
            <TremorButton
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center"
            >
              Delete User
            </TremorButton>
          </div>
        )}
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>Overview</Tab>
          <Tab>Raw JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {/* Overview Grid */}
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mb-6">
              <Card>
                <Text>User Role</Text>
                <div className="mt-2">
                  <Title>{userData.user_role || "Not Set"}</Title>
                </div>
              </Card>
              <Card>
                <Text>Team ID</Text>
                <div className="mt-2">
                  <Title>{userData.team_id || "Not Set"}</Title>
                </div>
              </Card>
              <Card>
                <Text>Budget</Text>
                <div className="mt-2">
                  <Text>Max Budget: ${userData.max_budget || "0"}</Text>
                  <Text>Current Spend: ${userData.spend || "0"}</Text>
                </div>
              </Card>
            </Grid>

            {/* Audit info shown as a subtle banner below the overview */}
            <div className="mb-6 text-sm text-gray-500 flex items-center gap-x-6">
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Created At {userData.created_at 
                  ? new Date(userData.created_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric'
                    })
                  : "Not Set"}
              </div>
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Last Updated {userData.last_updated_at 
                  ? new Date(userData.last_updated_at).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric'
                    })
                  : "Not Set"}
              </div>
            </div>

            {/* Settings Card */}
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>User Settings</Title>
                {canEditUser && !isEditing && (
                  <TremorButton
                    variant="secondary"
                    onClick={() => setIsEditing(true)}
                    className="flex items-center"
                  >
                    Edit User
                  </TremorButton>
                )}
              </div>
              <Form
                form={form}
                onFinish={handleUserUpdate}
                initialValues={{
                  user_id: localUserData.user_id,
                  user_email: localUserData.user_email,
                  user_role: localUserData.user_role,
                  max_budget: localUserData.max_budget,
                  spend: localUserData.spend,
                  team_id: localUserData.team_id,
                }}
                layout="vertical"
                onValuesChange={() => setIsDirty(true)}
              >
                <div className="space-y-4">
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">User ID</Text>
                      <div className="mt-1 p-2 bg-gray-50 rounded">{localUserData.user_id}</div>
                    </div>

                    <div>
                      <Text className="font-medium">Email</Text>
                      {isEditing ? (
                        <Form.Item name="user_email" className="mb-0">
                          <TextInput placeholder="Enter user email" />
                        </Form.Item>
                      ) : (
                        <div className="mt-1 p-2 bg-gray-50 rounded">{localUserData.user_email || "Not Set"}</div>
                      )}
                    </div>

                    <div>
                      <Text className="font-medium">Role</Text>
                      {isEditing ? (
                        <Form.Item name="user_role" className="mb-0">
                          <Select>
                            <Select.Option value="admin">Admin</Select.Option>
                            <Select.Option value="user">User</Select.Option>
                          </Select>
                        </Form.Item>
                      ) : (
                        <div className="mt-1 p-2 bg-gray-50 rounded">{localUserData.user_role || "Not Set"}</div>
                      )}
                    </div>

                    <div>
                      <Text className="font-medium">Max Budget</Text>
                      {isEditing ? (
                        <Form.Item name="max_budget" className="mb-0">
                          <TextInput placeholder="Enter max budget" />
                        </Form.Item>
                      ) : (
                        <div className="mt-1 p-2 bg-gray-50 rounded">${localUserData.max_budget || "0"}</div>
                      )}
                    </div>

                    <div>
                      <Text className="font-medium">Current Spend</Text>
                      {isEditing ? (
                        <Form.Item name="spend" className="mb-0">
                          <TextInput placeholder="Enter current spend" />
                        </Form.Item>
                      ) : (
                        <div className="mt-1 p-2 bg-gray-50 rounded">${localUserData.spend || "0"}</div>
                      )}
                    </div>

                    <div>
                      <Text className="font-medium">Team ID</Text>
                      {isEditing ? (
                        <Form.Item name="team_id" className="mb-0">
                          <TextInput placeholder="Enter team ID" />
                        </Form.Item>
                      ) : (
                        <div className="mt-1 p-2 bg-gray-50 rounded">{localUserData.team_id || "Not Set"}</div>
                      )}
                    </div>
                  </div>

                  {isEditing && (
                    <div className="mt-6 flex justify-end gap-2">
                      <TremorButton
                        variant="secondary"
                        onClick={() => {
                          form.resetFields();
                          setIsDirty(false);
                          setIsEditing(false);
                        }}
                      >
                        Cancel
                      </TremorButton>
                      <TremorButton
                        variant="primary"
                        onClick={() => form.submit()}
                        loading={isSaving}
                      >
                        Save Changes
                      </TremorButton>
                    </div>
                  )}
                </div>
              </Form>
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <pre className="bg-gray-100 p-4 rounded text-xs overflow-auto">
                {JSON.stringify(userData, null, 2)}
              </pre>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

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
                      <p className="text-sm font-medium text-gray-900 mt-2">
                        User ID: {userId}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button
                  onClick={handleDelete}
                  className="ml-2"
                  danger
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
    </div>
  );
} 