import { useQueryStates } from "nuqs";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import BulkEditUserModal from "./BulkEditUsers";
import BulkCreateUsersButton from "@/components/bulk_create_users_button";
import { CreateUserButton } from "@/components/CreateUserButton";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";
import { useUrlTab } from "@/hooks/useUrlTab";
import {
  getPossibleUserRoles,
  getProxyBaseUrl,
  invitationCreateCall,
  userListCall,
  UserListResponse,
} from "@/components/networking";
import OnboardingModal, { InvitationLink } from "@/components/onboarding_link";

import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { isAdminRole, isProxyAdminRole } from "@/utils/roles";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ColumnFiltersState,
  OnChangeFn,
  PaginationState,
  RowSelectionState,
  SortingState,
} from "@tanstack/react-table";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { modelAvailableCall, userDeleteCall } from "@/components/networking";
import { DefaultUserSettingsForm } from "./default-user-settings/DefaultUserSettingsForm";
import { UsersTable } from "./view_users/UsersTable";
import UserInfoView from "./view_users/user_info_view";
import { USER_DETAIL_URL_PARSERS } from "./view_users/useUserDetailUrlState";
import { UserInfo } from "@/components/networking";

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  teams: any[] | null;
  orgAdminOrgIds?: Array<{ organization_id: string; organization_alias: string }> | null;
}

type UserFilterColumn = "user_id" | "sso_user_id" | "user_role" | "team";

const USER_TABLE_STATE_OPTIONS: UrlTableStateOptions<UserFilterColumn> = {
  sortFields: ["user_id", "user_email", "user_role", "spend", "created_at"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: ["user_id", "sso_user_id", "user_role", "team"],
  urlKeys: {
    search: "user_search",
    filter_sso_user_id: "filter_sso_id",
    filter_user_role: "filter_role",
  },
};

const ADMIN_TABS = ["users", "default-settings"] as const;
type UsersPageTab = (typeof ADMIN_TABS)[number];
const MEMBER_TABS: readonly UsersPageTab[] = ["users"];

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  teams,
  orgAdminOrgIds,
}) => {
  const isProxyAdmin = userRole ? isProxyAdminRole(userRole) : false;
  const queryClient = useQueryClient();

  const {
    search: searchInput,
    setSearch,
    sorting,
    onSortingChange,
    pagination,
    onPaginationChange,
    columnFilters,
    onColumnFiltersChange,
  } = useUrlTableState(USER_TABLE_STATE_OPTIONS);
  const [searchQuery] = useDebouncedValue(searchInput, { wait: DEBOUNCE_WAIT_MS });
  const [activeTab, setActiveTab] = useUrlTab<UsersPageTab>(isProxyAdmin ? ADMIN_TABS : MEMBER_TABS, "users");

  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});
  const [selectionMode, setSelectionMode] = useState(false);
  const [isBulkEditModalVisible, setIsBulkEditModalVisible] = useState(false);

  const [{ user: selectedUserId }, setUserDetail] = useQueryStates(USER_DETAIL_URL_PARSERS);

  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDeletingUser, setIsDeletingUser] = useState(false);
  const [userToDelete, setUserToDelete] = useState<UserInfo | null>(null);
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false);
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null);
  const [baseUrl, setBaseUrl] = useState<string | null>(null);
  const [userModels, setUserModels] = useState<string[]>([]);

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
        setUserModels(available_model_names);
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const getFilterValue = useCallback(
    (columnId: string): string | undefined => {
      const entry = columnFilters.find((filter) => filter.id === columnId);
      return typeof entry?.value === "string" && entry.value.trim() ? entry.value.trim() : undefined;
    },
    [columnFilters],
  );

  const handleSearchChange = useCallback(
    (value: string) => {
      setSearch(value);
      setRowSelection({});
    },
    [setSearch],
  );

  const handleSortingChange = useCallback<OnChangeFn<SortingState>>(
    (updaterOrValue) => {
      onSortingChange(updaterOrValue);
      setRowSelection({});
    },
    [onSortingChange],
  );

  const handleColumnFiltersChange = useCallback<OnChangeFn<ColumnFiltersState>>(
    (updaterOrValue) => {
      onColumnFiltersChange(updaterOrValue);
      setRowSelection({});
    },
    [onColumnFiltersChange],
  );

  const handlePaginationChange = useCallback<OnChangeFn<PaginationState>>(
    (updaterOrValue) => {
      onPaginationChange(updaterOrValue);
      setRowSelection({});
    },
    [onPaginationChange],
  );

  const handleUserClick = useCallback(
    (userId: string, openInEdit: boolean = false) => {
      void setUserDetail(
        openInEdit ? { user: userId, user_tab: "details", edit: true } : { user: userId, user_tab: null, edit: null },
      );
    },
    [setUserDetail],
  );

  const handleCloseUserInfo = useCallback(() => {
    void setUserDetail(null);
  }, [setUserDetail]);

  const handleDelete = useCallback((user: UserInfo) => {
    setUserToDelete(user);
    setIsDeleteModalOpen(true);
  }, []);

  const handleResetPassword = useCallback(
    async (userId: string) => {
      if (!accessToken) {
        toast.fromError("Access token not found");
        return;
      }
      try {
        toast.success("Generating password reset link...");
        const data = await invitationCreateCall(accessToken, userId);
        setInvitationLinkData(data);
        setIsInvitationLinkModalVisible(true);
      } catch (error) {
        toast.fromError("Failed to generate password reset link");
      }
    },
    [accessToken],
  );

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

        toast.success("User deleted successfully");
      } catch (error) {
        console.error("Error deleting user:", error);
        toast.fromError("Failed to delete user");
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

  const handleToggleSelectionMode = () => {
    setSelectionMode(!selectionMode);
    setRowSelection({});
  };

  const handleBulkEditSuccess = () => {
    // Refresh the user list
    queryClient.invalidateQueries({ queryKey: ["userList"] });
    setRowSelection({});
    setSelectionMode(false);
  };

  const [activeSort] = sorting;
  const sortBy = activeSort.id;
  const sortOrder: "asc" | "desc" = activeSort.desc ? "desc" : "asc";

  const userIdFilter = getFilterValue("user_id");
  const ssoUserIdFilter = getFilterValue("sso_user_id");
  const userRoleFilter = getFilterValue("user_role");
  const teamFilter = getFilterValue("team");
  const searchFilter = searchQuery.trim() || null;

  const userListQueryFilters = {
    page: pagination.pageIndex + 1,
    pageSize: pagination.pageSize,
    search: searchFilter,
    userId: userIdFilter,
    ssoUserId: ssoUserIdFilter,
    role: userRoleFilter,
    team: teamFilter,
    sortBy,
    sortOrder,
    orgAdminOrgIds,
  };

  const userListQuery = useQuery({
    queryKey: ["userList", userListQueryFilters],
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required");

      return await userListCall(
        accessToken,
        userIdFilter ? [userIdFilter] : null,
        pagination.pageIndex + 1,
        pagination.pageSize,
        null,
        userRoleFilter ?? null,
        teamFilter ?? null,
        ssoUserIdFilter ?? null,
        sortBy,
        sortOrder,
        orgAdminOrgIds ? orgAdminOrgIds.map((o) => o.organization_id) : null,
        searchFilter,
      );
    },
    enabled: Boolean(accessToken && token && userRole && userID),
    placeholderData: (previousData) => previousData,
  });

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

  const users = useMemo<UserInfo[]>(() => userListQuery.data?.users ?? [], [userListQuery.data]);
  const totalUserCount = userListQuery.data?.total ?? 0;

  const selectedUsers = useMemo(() => users.filter((user) => rowSelection[user.user_id]), [users, rowSelection]);

  if (selectedUserId) {
    return (
      <UserInfoView
        userId={selectedUserId}
        onClose={handleCloseUserInfo}
        accessToken={accessToken}
        userRole={userRole}
        possibleUIRoles={possibleUIRoles}
      />
    );
  }

  const usersTable = (
    <UsersTable
      data={users}
      rowCount={totalUserCount}
      isLoading={userListQuery.isLoading || userListQuery.isPlaceholderData}
      isError={userListQuery.isError}
      possibleUIRoles={possibleUIRoles}
      teams={teams}
      sorting={sorting}
      onSortingChange={handleSortingChange}
      pagination={pagination}
      onPaginationChange={handlePaginationChange}
      columnFilters={columnFilters}
      onColumnFiltersChange={handleColumnFiltersChange}
      searchValue={searchInput}
      onSearchChange={handleSearchChange}
      selectionEnabled={isProxyAdmin && selectionMode}
      rowSelection={rowSelection}
      onRowSelectionChange={setRowSelection}
      onUserClick={handleUserClick}
      onDeleteUser={handleDelete}
      onResetPassword={handleResetPassword}
    />
  );

  return (
    <div className="w-full overflow-hidden p-8">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex space-x-3">
          {userListQuery.isLoading && (
            <>
              <Skeleton className="h-9 w-28" />
              <Skeleton className="h-9 w-36" />
              <Skeleton className="h-9 w-28" />
            </>
          )}
          {!userListQuery.isLoading && userID && accessToken && (
            <>
              {isProxyAdmin && (
                <CreateUserButton userID={userID} accessToken={accessToken} possibleUIRoles={possibleUIRoles} />
              )}

              {isProxyAdmin && (
                <BulkCreateUsersButton accessToken={accessToken} teams={teams} possibleUIRoles={possibleUIRoles} />
              )}

              {isProxyAdmin && (
                <Button
                  type="button"
                  onClick={handleToggleSelectionMode}
                  variant={selectionMode ? "default" : "outline"}
                  data-testid="toggle-user-selection"
                >
                  {selectionMode ? "Cancel Selection" : "Select Users"}
                </Button>
              )}

              {isProxyAdmin && selectionMode && (
                <Button
                  type="button"
                  onClick={() => setIsBulkEditModalVisible(true)}
                  disabled={selectedUsers.length === 0}
                  data-testid="bulk-edit-users"
                >
                  Bulk Edit ({selectedUsers.length} selected)
                </Button>
              )}
            </>
          )}
        </div>
      </div>

      {isProxyAdmin ? (
        <Tabs value={activeTab} onValueChange={setActiveTab} className="gap-0">
          <TabsList variant="line" className="mb-4">
            <TabsTrigger value="users" className="flex-none data-active:text-primary after:bg-primary">
              Users
            </TabsTrigger>
            <TabsTrigger value="default-settings" className="flex-none data-active:text-primary after:bg-primary">
              Default User Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="users" keepMounted>
            {usersTable}
          </TabsContent>

          <TabsContent value="default-settings" keepMounted>
            {!userID || !userRole || !accessToken ? (
              <div
                className="flex h-64 items-center justify-center"
                role="status"
                aria-label="Loading default user settings"
              >
                <div className="w-full max-w-lg space-y-3">
                  <Skeleton className="h-5 w-1/3" />
                  <Skeleton className="h-5 w-full" />
                  <Skeleton className="h-5 w-full" />
                  <Skeleton className="h-5 w-2/3" />
                </div>
              </div>
            ) : (
              <DefaultUserSettingsForm possibleUIRoles={possibleUIRoles} />
            )}
          </TabsContent>
        </Tabs>
      ) : (
        usersTable
      )}

      {/* Existing Modals */}
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
        open={isBulkEditModalVisible}
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
