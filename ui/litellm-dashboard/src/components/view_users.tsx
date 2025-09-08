import React, { useState, useEffect, useCallback, useRef } from "react"
import { Tab, TabGroup, TabList, TabPanels, TabPanel, Select, SelectItem } from "@tremor/react"

import { message } from "antd"

import {
  userInfoCall,
  userUpdateUserCall,
  getPossibleUserRoles,
  userListCall,
  UserListResponse,
  invitationCreateCall,
  getProxyBaseUrl,
} from "./networking"
import { Button } from "@tremor/react"
import CreateUser from "./create_user_button"
import EditUserModal from "./edit_user"
import OnboardingModal from "./onboarding_link"
import { InvitationLink } from "./onboarding_link"
import BulkEditUserModal from "./bulk_edit_user"

import { userDeleteCall, modelAvailableCall } from "./networking"
import { columns } from "./view_users/columns"
import { UserDataTable } from "./view_users/table"
import { UserInfo } from "./view_users/types"
import SSOSettings from "./SSOSettings"
import debounce from "lodash/debounce"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { updateExistingKeys } from "@/utils/dataUtils"
import { useDebouncedState } from "@tanstack/react-pacer/debouncer"
import { isAdminRole } from "@/utils/roles"
import NotificationsManager from "./molecules/notifications_manager"

interface ViewUserDashboardProps {
  accessToken: string | null
  token: string | null
  keys: any[] | null
  userRole: string | null
  userID: string | null
  teams: any[] | null
  setKeys: React.Dispatch<React.SetStateAction<Object[] | null>>
}

interface FilterState {
  email: string
  user_id: string
  user_role: string
  sso_user_id: string
  team: string
  model: string
  min_spend: number | null
  max_spend: number | null
  sort_by: string
  sort_order: "asc" | "desc"
}

const DEFAULT_PAGE_SIZE = 25

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
}

const ViewUserDashboard: React.FC<ViewUserDashboardProps> = ({ accessToken, token, userRole, userID, teams }) => {
  const queryClient = useQueryClient()
  const [currentPage, setCurrentPage] = useState(1)
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [selectedUser, setSelectedUser] = useState<UserInfo | null>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)
  const [userToDelete, setUserToDelete] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState("users")
  const [filters, setFilters] = useState<FilterState>(initialFilters)
  const [debouncedFilters, setDebouncedFilters, debouncer] = useDebouncedState(filters, { wait: 300 })
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false)
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null)
  const [baseUrl, setBaseUrl] = useState<string | null>(null)
  const [selectedUsers, setSelectedUsers] = useState<UserInfo[]>([])
  const [isBulkEditModalVisible, setIsBulkEditModalVisible] = useState(false)
  const [selectionMode, setSelectionMode] = useState(false)
  const [userModels, setUserModels] = useState<string[]>([])

  const handleDelete = (userId: string) => {
    setUserToDelete(userId)
    setIsDeleteModalOpen(true)
  }

  useEffect(() => {
    return () => {
      debouncer.cancel()
    }
  }, [debouncer])

  useEffect(() => {
    setBaseUrl(getProxyBaseUrl())
  }, [])

  // Fetch available models for bulk edit
  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (!userID || !userRole || !accessToken) {
          return
        }

        const model_available = await modelAvailableCall(accessToken, userID, userRole)
        let available_model_names = model_available["data"].map(
          (element: { id: string }) => element.id
        )
        console.log("available_model_names:", available_model_names)
        setUserModels(available_model_names)
      } catch (error) {
        console.error("Error fetching user models:", error)
      }
    }

    fetchUserModels()
  }, [accessToken, userID, userRole])

  const updateFilters = (update: Partial<FilterState>) => {
    setFilters((previousFilters) => {
      const newFilters = { ...previousFilters, ...update }
      setDebouncedFilters(newFilters)
      return newFilters
    })
  }

  const handleSortChange = (sortBy: string, sortOrder: "asc" | "desc") => {
    updateFilters({ sort_by: sortBy, sort_order: sortOrder })
  }

  const handleResetPassword = async (userId: string) => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token not found")
      return
    }
    try {
      NotificationsManager.success("Generating password reset link...")
      const data = await invitationCreateCall(accessToken, userId)
      setInvitationLinkData(data)
      setIsInvitationLinkModalVisible(true)
    } catch (error) {
      NotificationsManager.fromBackend("Failed to generate password reset link")
    }
  }

  const confirmDelete = async () => {
    if (userToDelete && accessToken) {
      try {
        await userDeleteCall(accessToken, [userToDelete])

        // Update the user list after deletion
        queryClient.setQueriesData<UserListResponse>({ queryKey: ["userList"] }, (previousData) => {
          if (previousData === undefined) return previousData
          const updatedUsers = previousData.users.filter((user) => user.user_id !== userToDelete)
          return { ...previousData, users: updatedUsers }
        })

        NotificationsManager.success("User deleted successfully")
      } catch (error) {
        console.error("Error deleting user:", error)
        NotificationsManager.fromBackend("Failed to delete user")
      }
    }
    setIsDeleteModalOpen(false)
    setUserToDelete(null)
  }

  const cancelDelete = () => {
    setIsDeleteModalOpen(false)
    setUserToDelete(null)
  }

  const handleEditCancel = async () => {
    setSelectedUser(null)
    setEditModalVisible(false)
  }

  const handleEditSubmit = async (editedUser: any) => {
    console.log("inside handleEditSubmit:", editedUser)

    if (!accessToken || !token || !userRole || !userID) {
      return
    }

    try {
      const response = await userUpdateUserCall(accessToken, editedUser, null)
      queryClient.setQueriesData<UserListResponse>({ queryKey: ["userList"] }, (previousData) => {
        if (previousData === undefined) return previousData
        const updatedUsers = previousData.users.map((user) => {
          if (user.user_id === response.data.user_id) {
            return updateExistingKeys(user, response.data)
          }
          return user
        })

        return { ...previousData, users: updatedUsers }
      })

      NotificationsManager.success(`User ${editedUser.user_id} updated successfully`)
    } catch (error) {
      console.error("There was an error updating the user", error)
    }
    setSelectedUser(null)
    setEditModalVisible(false)
    // Close the modal
  }

  const handlePageChange = async (newPage: number) => {
    setCurrentPage(newPage)
  }

  const handleToggleSelectionMode = () => {
    setSelectionMode(!selectionMode)
    setSelectedUsers([])
  }

  const handleSelectionChange = (users: UserInfo[]) => {
    setSelectedUsers(users)
  }

  const handleBulkEdit = () => {
    if (selectedUsers.length === 0) {
      NotificationsManager.fromBackend("Please select users to edit")
      return
    }

    setIsBulkEditModalVisible(true)
  }

  const handleBulkEditSuccess = () => {
    // Refresh the user list
    queryClient.invalidateQueries({ queryKey: ["userList"] })
    setSelectedUsers([])
    setSelectionMode(false)
  }

  const userListQuery = useQuery({
    queryKey: ["userList", { debouncedFilter: debouncedFilters, currentPage }],
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required")

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
      )
    },
    enabled: Boolean(accessToken && token && userRole && userID),
    placeholderData: (previousData) => previousData,
  })
  const userListResponse = userListQuery.data

  const userRolesQuery = useQuery<Record<string, Record<string, string>>>({
    queryKey: ["userRoles"],
    initialData: () => ({}),
    queryFn: async () => {
      if (!accessToken) throw new Error("Access token required")
      return await getPossibleUserRoles(accessToken)
    },
    enabled: Boolean(accessToken && token && userRole && userID),
  })
  const possibleUIRoles = userRolesQuery.data

  if (userListQuery.isLoading) {
    return <div>Loading...</div>
  }

  if (!accessToken || !token || !userRole || !userID) {
    return <div>Loading...</div>
  }

  const tableColumns = columns(
    possibleUIRoles,
    (user) => {
      setSelectedUser(user)
      setEditModalVisible(true)
    },
    handleDelete,
    handleResetPassword,
    () => {}, // placeholder function, will be overridden in UserDataTable
  )

  return (
    <div className="w-full p-8 overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div className="flex space-x-3">
          <CreateUser userID={userID} accessToken={accessToken} teams={teams} possibleUIRoles={possibleUIRoles} />
          
          <Button
            onClick={handleToggleSelectionMode}
            variant={selectionMode ? "primary" : "secondary"}
            className="flex items-center"
          >
            {selectionMode ? "Cancel Selection" : "Select Users"}
          </Button>
          
          {selectionMode && (
            <Button
              onClick={handleBulkEdit}
              disabled={selectedUsers.length === 0}
              className="flex items-center"
            >
              Bulk Edit ({selectedUsers.length} selected)
            </Button>
          )}
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
                setSelectedUser(user)
                setEditModalVisible(true)
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
            <SSOSettings
              accessToken={accessToken}
              possibleUIRoles={possibleUIRoles}
              userID={userID}
              userRole={userRole}
            />
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
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            {/* Modal Panel */}
            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">
              &#8203;
            </span>

            {/* Confirmation Modal Content */}
            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Delete User</h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">Are you sure you want to delete this user?</p>
                      <p className="text-sm font-medium text-gray-900 mt-2">User ID: {userToDelete}</p>
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
  )
}

export default ViewUserDashboard
