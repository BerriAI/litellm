import React, { useState } from "react"
import { Card, Text, Button, Grid, Col, Tab, TabList, TabGroup, TabPanel, TabPanels, Title, Badge } from "@tremor/react"
import { ArrowLeftIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline"
import {
  userInfoCall,
  userDeleteCall,
  userUpdateUserCall,
  modelAvailableCall,
  invitationCreateCall,
  getProxyBaseUrl,
} from "../networking"
import { message, Button as AntdButton } from "antd"
import { rolesWithWriteAccess } from "../../utils/roles"
import { UserEditView } from "../user_edit_view"
import OnboardingModal, { InvitationLink } from "../onboarding_link"
import { formatNumberWithCommas, copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils"
import { CopyIcon, CheckIcon } from "lucide-react"
import NotificationsManager from "../molecules/notifications_manager"
import { getBudgetDurationLabel } from "../common_components/budget_duration_dropdown"

interface UserInfoViewProps {
  userId: string
  onClose: () => void
  accessToken: string | null
  userRole: string | null
  onDelete?: () => void
  possibleUIRoles: Record<string, Record<string, string>> | null
  initialTab?: number // 0 for Overview, 1 for Details
  startInEditMode?: boolean
}

interface UserInfo {
  user_id: string
  user_info: {
    user_email: string | null
    user_role: string | null
    teams: any[] | null
    models: string[] | null
    max_budget: number | null
    budget_duration: string | null
    spend: number | null
    metadata: Record<string, any> | null
    created_at: string | null
    updated_at: string | null
  }
  keys: any[] | null
  teams: any[] | null
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
  const [userData, setUserData] = useState<UserInfo | null>(null)
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isEditing, setIsEditing] = useState(startInEditMode)
  const [userModels, setUserModels] = useState<string[]>([])
  const [isInvitationLinkModalVisible, setIsInvitationLinkModalVisible] = useState(false)
  const [invitationLinkData, setInvitationLinkData] = useState<InvitationLink | null>(null)
  const [baseUrl, setBaseUrl] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState(initialTab)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})
  const [isTeamsExpanded, setIsTeamsExpanded] = useState(false)

  React.useEffect(() => {
    setBaseUrl(getProxyBaseUrl())
  }, [])

  React.useEffect(() => {
    console.log(`userId: ${userId}, userRole: ${userRole}, accessToken: ${accessToken}`)
    const fetchData = async () => {
      try {
        if (!accessToken) return
        const data = await userInfoCall(accessToken, userId, userRole || "", false, null, null, true)
        setUserData(data)

        // Fetch available models
        const modelDataResponse = await modelAvailableCall(accessToken, userId, userRole || "")
        const availableModels = modelDataResponse.data.map((model: any) => model.id)
        setUserModels(availableModels)
      } catch (error) {
        console.error("Error fetching user data:", error)
        NotificationsManager.fromBackend("Failed to fetch user data")
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [accessToken, userId, userRole])

  const handleResetPassword = async () => {
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

  const handleDelete = async () => {
    try {
      if (!accessToken) return
      await userDeleteCall(accessToken, [userId])
      NotificationsManager.success("User deleted successfully")
      if (onDelete) {
        onDelete()
      }
      onClose()
    } catch (error) {
      console.error("Error deleting user:", error)
      NotificationsManager.fromBackend("Failed to delete user")
    }
  }

  const handleUserUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken || !userData) return

      const response = await userUpdateUserCall(accessToken, formValues, null)

      // Update local state with new values
      setUserData({
        ...userData,
        user_info: {
          ...userData.user_info,
          user_email: formValues.user_email,
          models: formValues.models,
          max_budget: formValues.max_budget,
          budget_duration: formValues.budget_duration,
          metadata: formValues.metadata,
        },
      })

      NotificationsManager.success("User updated successfully")
      setIsEditing(false)
    } catch (error) {
      console.error("Error updating user:", error)
      NotificationsManager.fromBackend("Failed to update user")
    }
  }

  if (isLoading) {
    return (
      <div className="p-4">
        <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Users
        </Button>
        <Text>Loading user data...</Text>
      </div>
    )
  }

  if (!userData) {
    return (
      <div className="p-4">
        <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Users
        </Button>
        <Text>User not found</Text>
      </div>
    )
  }

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text)
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }))
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }))
      }, 2000)
    }
  }

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
              className="flex items-center"
            >
              Delete User
            </Button>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">
              &#8203;
            </span>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">Delete User</h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">Are you sure you want to delete this user?</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <Button onClick={handleDelete} color="red" className="ml-2">
                  Delete
                </Button>
                <Button onClick={() => setIsDeleteModalOpen(false)}>Cancel</Button>
              </div>
            </div>
          </div>
        </div>
      )}

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
                <Text>API Keys</Text>
                <div className="mt-2">
                  <Text>{userData.keys?.length || 0} keys</Text>
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
                    <Text className="font-medium">API Keys</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {userData.keys?.length && userData.keys?.length > 0 ? (
                        userData.keys.map((key, index) => (
                          <span key={index} className="px-2 py-1 bg-green-100 rounded text-xs">
                            {key.key_alias || key.token}
                          </span>
                        ))
                      ) : (
                        <Text>No API keys</Text>
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
  )
}
