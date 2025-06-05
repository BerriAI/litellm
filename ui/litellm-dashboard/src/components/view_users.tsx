import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Tab,
  TabGroup,
  TabList,
  TabPanels,
  TabPanel,
  Select,
  SelectItem,
} from "@tremor/react";

import { message } from "antd";

import {
  userInfoCall,
  userUpdateUserCall,
  getPossibleUserRoles,
  userListCall,
  UserListResponse,
} from "./networking";
import { Button } from "@tremor/react";
import CreateUser from "./create_user_button";
import EditUserModal from "./edit_user";

import { userDeleteCall } from "./networking";
import { columns } from "./view_users/columns";
import { UserDataTable } from "./view_users/table";
import { UserInfo } from "./view_users/types";
import SSOSettings from "./SSOSettings";
import debounce from "lodash/debounce";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { updateExistingKeys } from "@/utils/dataUtils";
import { useDebouncedState } from '@tanstack/react-pacer/debouncer'

interface ViewUserDashboardProps {
  accessToken: string | null;
  token: string | null;
  keys: any[] | null;
  userRole: string | null;
  userID: string | null;
  teams: any[] | null;
  setKeys: React.Dispatch<React.SetStateAction<Object[] | null>>;
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
  sort_order: 'asc' | 'desc';
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
  sort_order: "desc"
}

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  teams,
}) => {
  const queryClient = useQueryClient();
  const [currentPage, setCurrentPage] = useState(1);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [userToDelete, setUserToDelete] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("users");
  const [filters, setFilters] = useState<FilterState>(initialFilters);
  const [debouncedFilters, setDebouncedFilters, debouncer] = useDebouncedState(filters, { wait: 300 })
  const [showFilters, setShowFilters] = useState(false);
  const handleDelete = (userId: string) => {
    setUserToDelete(userId);
    setIsDeleteModalOpen(true);
  };

  useEffect(() => {
    return () => {
      debouncer.cancel()
    }
  }, [debouncer])

  const updateFilters = (update: Partial<FilterState>) => {
    setFilters((previousFilters) => {
      const newFilters = {...previousFilters, ...update };
      setDebouncedFilters(newFilters);
      return newFilters;
    })
  };

  const handleSortChange = (sortBy: string, sortOrder: 'asc' | 'desc') => {
    updateFilters({ sort_by: sortBy, sort_order: sortOrder });
  };


  const confirmDelete = async () => {
    if (userToDelete && accessToken) {
      try {
        await userDeleteCall(accessToken, [userToDelete]);

        // Update the user list after deletion
        queryClient.setQueriesData<UserListResponse>({ queryKey: ['userList'] }, (previousData) => {
          if (previousData === undefined) return previousData;
          const updatedUsers = previousData.users.filter(user => user.user_id !== userToDelete);
          return { ...previousData, users: updatedUsers };
        })
        
        message.success("User deleted successfully");
      } catch (error) {
        console.error("Error deleting user:", error);
        message.error("Failed to delete user");
      }
    }
    setIsDeleteModalOpen(false);
    setUserToDelete(null);
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
      queryClient.setQueriesData<UserListResponse>({ queryKey: ['userList'] }, (previousData) => {
        if (previousData === undefined) return previousData;
        const updatedUsers = previousData.users.map(user => {
          if (user.user_id === response.data.user_id) {
            return updateExistingKeys(user, response.data);
          }
          return user;
        });
        
        return { ...previousData, users: updatedUsers };
      })

      message.success(`User ${editedUser.user_id} updated successfully`);
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

  const userListQuery = useQuery({
    queryKey: ['userList', { debouncedFilter: debouncedFilters, currentPage }],
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');

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
        debouncedFilters.sort_order
      );
    },
    enabled: Boolean(accessToken && token && userRole && userID),
    placeholderData: (previousData) => previousData
  });
  const userListResponse = userListQuery.data

  const userRolesQuery = useQuery<Record<string, Record<string, string>>>({
    queryKey: ['userRoles'],
    initialData: () => ({}),
    queryFn: async () => {
      if (!accessToken) throw new Error('Access token required');
      return await getPossibleUserRoles(accessToken);
    },
    enabled: Boolean(accessToken && token && userRole && userID),
  });
  const possibleUIRoles = userRolesQuery.data

  if (userListQuery.isLoading) {
    return <div>Loading...</div>
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>;
  }

  const tableColumns = columns(
    possibleUIRoles,
    (user) => {
      setSelectedUser(user);
      setEditModalVisible(true);
    },
    handleDelete
  );

  return (
    <div className="w-full p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">Users</h1>
        <div className="flex space-x-3">
          <CreateUser
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
          />
        </div>
      </div>
      
      <TabGroup defaultIndex={0} onIndexChange={(index) => setActiveTab(index === 0 ? "users" : "settings")}>
        <TabList className="mb-4">
          <Tab>Users</Tab>
          <Tab>Default User Settings</Tab>
        </TabList>
        
        <TabPanels>
          <TabPanel>
            <div className="bg-white rounded-lg shadow">
              <div className="border-b px-6 py-4">
                <div className="flex flex-col space-y-4">
                  {/* Search and Filter Controls */}
                  <div className="flex flex-wrap items-center gap-3">
                    {/* Email Search */}
                    <div className="relative w-64">
                      <input
                        type="text"
                        placeholder="Search by email..."
                        className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        value={filters.email}
                        onChange={(e) => updateFilters({ email: e.target.value })}
                      />
                      <svg
                        className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                        />
                      </svg>
                    </div>

                    {/* Filter Button */}
                    <button
                      className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? 'bg-gray-100' : ''}`}
                      onClick={() => setShowFilters(!showFilters)}
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                        />
                      </svg>
                      Filters
                      {(filters.user_id || filters.user_role || filters.team) && (
                        <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                      )}
                    </button>

                    {/* Reset Filters Button */}
                    <button
                      className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                      onClick={() => {
                        updateFilters(initialFilters);
                      }}
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                        />
                      </svg>
                      Reset Filters
                    </button>
                  </div>

                  {/* Additional Filters */}
                  {showFilters && (
                    <div className="flex flex-wrap items-center gap-3 mt-3">
                      {/* User ID Search */}
                      <div className="relative w-64">
                        <input
                          type="text"
                          placeholder="Filter by User ID"
                          className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          value={filters.user_id}
                          onChange={(e) => updateFilters({ user_id : e.target.value })}
                        />
                        <svg
                          className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5.121 17.804A13.937 13.937 0 0112 16c2.5 0 4.847.655 6.879 1.804M15 10a3 3 0 11-6 0 3 3 0 016 0zm6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                      </div>

                      {/* Role Dropdown */}
                      <div className="w-64">
                        <Select
                          value={filters.user_role}
                          onValueChange={(value) => updateFilters({ user_role: value })}
                          placeholder="Select Role"
                        >
                          {Object.entries(possibleUIRoles).map(([key, value]) => (
                            <SelectItem key={key} value={key}>
                              {value.ui_label}
                            </SelectItem>
                          ))}
                        </Select>
                      </div>

                      {/* Team Dropdown */}
                      <div className="w-64">
                        <Select
                          value={filters.team}
                          onValueChange={(value) => updateFilters({ team: value })}
                          placeholder="Select Team"
                        >
                          {teams?.map((team) => (
                            <SelectItem key={team.team_id} value={team.team_id}>
                              {team.team_alias || team.team_id}
                            </SelectItem>
                          ))}
                        </Select>
                      </div>
                      
                      {/* SSO ID Search */}
                      <div className="relative w-64">
                        <input
                          type="text"
                          placeholder="Filter by SSO ID"
                          className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                          value={filters.sso_user_id}
                          onChange={(e) => updateFilters({ sso_user_id : e.target.value })}
                        />
                      </div>
                    </div>
                  )}

                  {/* Results Count and Pagination */}
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">
                      Showing{" "}
                      {userListResponse && userListResponse.users && userListResponse.users.length > 0
                        ? (userListResponse.page - 1) * userListResponse.page_size + 1
                        : 0}{" "}
                      -{" "}
                      {userListResponse && userListResponse.users
                        ? Math.min(
                            userListResponse.page * userListResponse.page_size,
                            userListResponse.total
                          )
                        : 0}{" "}
                      of {userListResponse ? userListResponse.total : 0} results
                    </span>
                    
                    {/* Pagination Buttons */}
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handlePageChange(currentPage - 1)}
                        disabled={currentPage === 1}
                        className={`px-3 py-1 text-sm border rounded-md ${
                          currentPage === 1
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'hover:bg-gray-50'
                        }`}
                      >
                        Previous
                      </button>
                      <button
                        onClick={() => handlePageChange(currentPage + 1)}
                        disabled={!userListResponse || currentPage >= userListResponse.total_pages}
                        className={`px-3 py-1 text-sm border rounded-md ${
                          !userListResponse || currentPage >= userListResponse.total_pages
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'hover:bg-gray-50'
                        }`}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <UserDataTable
                data={userListQuery.data?.users || []}
                columns={tableColumns}
                isLoading={userListQuery.isLoading}
                accessToken={accessToken}
                userRole={userRole}
                onSortChange={handleSortChange}
                currentSort={{
                  sortBy: filters.sort_by,
                  sortOrder: filters.sort_order
                }}
                possibleUIRoles={possibleUIRoles}
              />
            </div>
          </TabPanel>
          
          <TabPanel>
            <SSOSettings accessToken={accessToken} possibleUIRoles={possibleUIRoles} userID={userID} userRole={userRole}/>
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

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div
              className="fixed inset-0 transition-opacity"
              aria-hidden="true"
            >
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            {/* Modal Panel */}
            <span
              className="hidden sm:inline-block sm:align-middle sm:h-screen"
              aria-hidden="true"
            >
              &#8203;
            </span>

            {/* Confirmation Modal Content */}
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
                        User ID: {userToDelete}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={confirmDelete} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={cancelDelete}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ViewUserDashboard;
