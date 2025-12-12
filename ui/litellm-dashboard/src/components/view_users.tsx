import { Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import React, { useEffect, useState } from "react";

import { Button } from "@tremor/react";
import BulkEditUserModal from "./bulk_edit_user";
import CreateUser from "./create_user_button";
import EditUserModal from "./edit_user";
import {
  getPossibleUserRoles,
  getProxyBaseUrl,
  invitationCreateCall,
  userListCall,
  UserListResponse,
  userUpdateUserCall,
} from "./networking";
import OnboardingModal, { InvitationLink } from "./onboarding_link";

import { updateExistingKeys } from "@/utils/dataUtils";
import { isAdminRole } from "@/utils/roles";
import { useDebouncedState } from "@tanstack/react-pacer/debouncer";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Typography } from "antd";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import NotificationsManager from "./molecules/notifications_manager";
import { modelAvailableCall, userDeleteCall } from "./networking";
import DefaultUserSettings from "./DefaultUserSettings";
import { columns } from "./view_users/columns";
import { UserDataTable } from "./view_users/table";
import { UserInfo } from "./view_users/types";
import { Skeleton } from "antd";

const { Text, Title } = Typography;

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  keys: any[] | null;
  userRole: string | null;
  userID: string | null;
  teams: any[] | null;
  setKeys: React.Dispatch<React.SetStateAction<object[] | null>>;
}

interface FilterState {
  email: string;
  user_id: string;
  user_role: string;
  sso_user_id: string;
  team: string;
  model: string;
  min_spend: number | null;
  max_spend: number | null;
  sort_by: string;
  sort_order: "asc" | "desc";
}

const DEFAULT_PAGE_SIZE = 25;

const initialFilters: FilterState = {
  email: "",
  user_id: "",
  user_role: "",
  sso_user_id: "",
  team: "",
  model: "",
  min_spend: null,
  max_spend: null,
  sort_by: "created_at",
  sort_order: "desc",
};

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({ accessToken, token, userRole, userID, teams }) => {
  const queryClient = useQueryClient();
  const [currentPage, setCurrentPage] = useState(1);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeletingUser, setIsDeletingUser] = useState(false);
  const [userToDelete, setUserToDelete] = useState<UserInfo | null>(null);
  const [activeTab, setActiveTab] = useState("users");
  const [filters, setFilters] = useState<FilterState>(initialFilters);
  const [debouncedFilters, setDebouncedFilters, debouncer] = useDebouncedState(filters, { wait: 300 });
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [selectedUsers, setSelectedUsers] = useState<UserInfo[]>([]);
  const [isBulkEditModalVisible, setIsBulkEditModalVisible] = useState(false);
  const [selectionMode, setSelectionMode] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);

  const handleDelete = (user: UserInfo) => {
    setUserToDelete(user);
    setIsDeleteModalOpen(true);
  };

  useEffect(() => {
    return () => {
      debouncer.cancel();
    };
  }, [debouncer]);

  useEffect(() => {
    setBaseUrl(getProxyBaseUrl());
  }, []);

  // Fetch available models for bulk edit
  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (!userID || !userRole || !accessToken) {
          return;
        }

        const model_available = await modelAvailableCall(accessToken, userID, userRole);
        let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
        console.log("available_model_names:", available_model_names);
        setUserModels(available_model_names);
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const updateFilters = (update: Partial<FilterState>) => {
    setFilters((previousFilters) => {
      const newFilters = { ...previousFilters, ...update };
      setDebouncedFilters(newFilters);
      return newFilters;
    });
  };

  const handleSortChange = (sortBy: string, sortOrder: "asc" | "desc") => {
    updateFilters({ sort_by: sortBy, sort_order: sortOrder });
  };

  const handleResetPassword = async (userId: string) => {
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

  const confirmDelete = async () => {
    if (userToDelete && accessToken) {
      try {
        setIsDeletingUser(true);
        await userDeleteCall(accessToken, [userToDelete.user_id]);

        // Update the user list after deletion
        queryClient.setQueriesData<UserListResponse>({ queryKey: ["userList"] }, (previousData) => {
          if (previousData === undefined) return previousData;
          const updatedUsers = previousData.users.filter((user) => user.user_id !== userToDelete.user_id);
          return { ...previousData, users: updatedUsers };
        });

        NotificationsManager.success("User deleted successfully");
      } catch (error) {
        console.error("Error deleting user:", error);
        NotificationsManager.fromBackend("Failed to delete user");
      } finally {
        setIsDeleteModalOpen(false);
        setUserToDelete(null);
        setIsDeletingUser(false);
      }
    }
  };

  const cancelDelete = () => {
    setIsDeleteModalOpen(false);
    setUserToDelete(null);
  };

  const handleEditCancel = async () => {
    setSelectedUser(null);
    setEditModalVisible(false);
  };

  const handleEditSubmit = async (editedUser: any) => {
    console.log("inside handleEditSubmit:", editedUser);

    if (!accessToken || !token || !userRole || !userID) {
      return;
    }

    try {
      const response = await userUpdateUserCall(accessToken, editedUser, null);
      queryClient.setQueriesData<UserListResponse>({ queryKey: ["userList"] }, (previousData) => {
        if (previousData === undefined) return previousData;
        const updatedUsers = previousData.users.map((user) => {
          if (user.user_id === response.data.user_id) {
            return updateExistingKeys(user, response.data);
          }
          return user;
        });

        return { ...previousData, users: updatedUsers };
      });

      NotificationsManager.success(`User ${editedUser.user_id} updated successfully`);
    } catch (error) {
      console.error("There was an error updating the user", error);
    }
    setSelectedUser(null);
    setEditModalVisible(false);
    // Close the modal
  };

  const handlePageChange = async (newPage: number) => {
    setCurrentPage(newPage);
  };

  const handleToggleSelectionMode = () => {
    setSelectionMode(!selectionMode);
    setSelectedUsers([]);
  };

  const handleSelectionChange = (users: UserInfo[]) => {
    setSelectedUsers(users);
  };

  const handleBulkEdit = () => {
    if (selectedUsers.length === 0) {
      NotificationsManager.fromBackend("Please select users to edit");
      return;
    }

    setIsBulkEditModalVisible(true);
  };

  const handleBulkEditSuccess = () => {
    // Refresh the user list
    queryClient.invalidateQueries({ queryKey: ["userList"] });
    setSelectedUsers([]);
    setSelectionMode(false);
  };

  const userListQuery = useQuery({
    queryKey: ["userList", { debouncedFilter: debouncedFilters, currentPage }],
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required");

      return await userListCall(
        accessToken,
        debouncedFilters.user_id ? [debouncedFilters.user_id] : null,
        currentPage,
        DEFAULT_PAGE_SIZE,
        debouncedFilters.email || null,
        debouncedFilters.user_role || null,
        debouncedFilters.team || null,
        debouncedFilters.sso_user_id || null,
        debouncedFilters.sort_by,
        debouncedFilters.sort_order,
      );
    },
    enabled: Boolean(accessToken && token && userRole && userID),
    placeholderData: (previousData) => previousData,
  });
  const userListResponse = userListQuery.data;

  const userRolesQuery = useQuery<Record<string, Record<string, string>>>({
    queryKey: ["userRoles"],
    initialData: () => ({}),
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required");
      return await getPossibleUserRoles(accessToken);
    },
    enabled: Boolean(accessToken && token && userRole && userID),
  });
  const possibleUIRoles = userRolesQuery.data;

  const tableColumns = columns(
    possibleUIRoles,
    (user) => {
      setSelectedUser(user);
      setEditModalVisible(true);
    },
    handleDelete,
    handleResetPassword,
    () => {}, // placeholder function, will be overridden in UserDataTable
  );

  return (
    <div className="w-full p-8 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="flex space-x-3">
          {userListQuery.isLoading ? (
            <>
              <Skeleton.Button active size="default" shape="default" style={{ width: 110, height: 36 }} />
              <Skeleton.Button active size="default" shape="default" style={{ width: 145, height: 36 }} />
              <Skeleton.Button active size="default" shape="default" style={{ width: 110, height: 36 }} />
            </>
          ) : userID && accessToken ? (
            <>
              <CreateUser userID={userID} accessToken={accessToken} teams={teams} possibleUIRoles={possibleUIRoles} />

              <Button
                onClick={handleToggleSelectionMode}
                variant={selectionMode ? "primary" : "secondary"}
                className="flex items-center"
              >
                {selectionMode ? "Cancel Selection" : "Select Users"}
              </Button>

              {selectionMode && (
                <Button onClick={handleBulkEdit} disabled={selectedUsers.length === 0} className="flex items-center">
                  Bulk Edit ({selectedUsers.length} selected)
                </Button>
              )}
            </>
          ) : null}
        </div>
      </div>

      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(index === 0 ? "users" : "settings")}>
        <TabList className="mb-4">
          <Tab>Users</Tab>
          <Tab>Default User Settings</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <UserDataTable
              data={userListQuery.data?.users || []}
              columns={tableColumns}
              isLoading={userListQuery.isLoading}
              accessToken={accessToken}
              userRole={userRole}
              onSortChange={handleSortChange}
              currentSort={{
                sortBy: filters.sort_by,
                sortOrder: filters.sort_order,
              }}
              possibleUIRoles={possibleUIRoles}
              handleEdit={(user) => {
                setSelectedUser(user);
                setEditModalVisible(true);
              }}
              handleDelete={handleDelete}
              handleResetPassword={handleResetPassword}
              enableSelection={selectionMode}
              selectedUsers={selectedUsers}
              onSelectionChange={handleSelectionChange}
              filters={filters}
              updateFilters={updateFilters}
              initialFilters={initialFilters}
              teams={teams}
              userListResponse={userListResponse}
              currentPage={currentPage}
              handlePageChange={handlePageChange}
            />
          </TabPanel>

          <TabPanel>
            {!userID || !userRole || !accessToken ? (
              <div className="flex justify-center items-center h-64">
                <Skeleton active paragraph={{ rows: 4 }} />
              </div>
            ) : (
              <DefaultUserSettings
                accessToken={accessToken}
                possibleUIRoles={possibleUIRoles}
                userID={userID}
                userRole={userRole}
              />
            )}
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Existing Modals */}
      <EditUserModal
        visible={editModalVisible}
        possibleUIRoles={possibleUIRoles}
        onCancel={handleEditCancel}
        user={selectedUser}
        onSubmit={handleEditSubmit}
      />

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete User?"
        message="Are you sure you want to delete this user? This action cannot be undone."
        resourceInformationTitle="User Information"
        resourceInformation={[
          { label: "Email", value: userToDelete?.user_email },
          { label: "User ID", value: userToDelete?.user_id, code: true },
          {
            label: "Global Proxy Role",
            value:
              (userToDelete && possibleUIRoles?.[userToDelete.user_role]?.ui_label) || userToDelete?.user_role || "-",
          },
          { label: "Total Spend (USD)", value: userToDelete?.spend?.toFixed(2) },
        ]}
        onCancel={cancelDelete}
        onOk={confirmDelete}
        confirmLoading={isDeletingUser}
      />

      <OnboardingModal
        isInvitationLinkModalVisible={isInvitationLinkModalVisible}
        setIsInvitationLinkModalVisible={setIsInvitationLinkModalVisible}
        baseUrl={baseUrl || ""}
        invitationLinkData={invitationLinkData}
        modalType="resetPassword"
      />

      <BulkEditUserModal
        visible={isBulkEditModalVisible}
        onCancel={() => setIsBulkEditModalVisible(false)}
        selectedUsers={selectedUsers}
        possibleUIRoles={possibleUIRoles}
        accessToken={accessToken}
        onSuccess={handleBulkEditSuccess}
        teams={teams}
        userRole={userRole}
        userModels={userModels}
        allowAllUsers={userRole ? isAdminRole(userRole) : false}
      />
    </div>
  );
};

export default ViewUserDashboard;
