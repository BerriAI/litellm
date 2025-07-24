import React, { useEffect, useState } from "react";
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
  TextInput,
  Select as TremorSelect
} from "@tremor/react";
import { ArrowLeftIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import { keyDeleteCall, keyUpdateCall } from "./networking";
import { KeyResponse } from "./key_team_helpers/key_list";
import { Form, Input, InputNumber, message, Select, Tooltip, Button as AntdButton } from "antd";
import { KeyEditView } from "./key_edit_view";
import { RegenerateKeyModal } from "./regenerate_key_modal";
import { rolesWithWriteAccess } from '../utils/roles';
import ObjectPermissionsView from "./object_permissions_view";
import LoggingSettingsView from "./logging_settings_view";
import { copyToClipboard as utilCopyToClipboard, formatNumberWithCommas } from "@/utils/dataUtils";
import { extractLoggingSettings, formatMetadataForDisplay } from "./key_info_utils";
import { CopyIcon, CheckIcon } from "lucide-react";
import { callback_map, mapInternalToDisplayNames, mapDisplayToInternalNames } from "./callback_info_helpers";

interface KeyInfoViewProps {
  keyId: string
  onClose: () => void
  keyData: KeyResponse | undefined
  onKeyDataUpdate?: (data: Partial<KeyResponse>) => void
  onDelete?: () => void
  accessToken: string | null
  userID: string | null
  userRole: string | null
  teams: any[] | null
  premiumUser: boolean
  setAccessToken?: (token: string) => void
}

export default function KeyInfoView({
  keyId,
  onClose,
  keyData,
  accessToken,
  userID,
  userRole,
  teams,
  onKeyDataUpdate,
  onDelete,
  premiumUser,
  setAccessToken,
}: KeyInfoViewProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [form] = Form.useForm()
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false)
  const [isRegenerateModalOpen, setIsRegenerateModalOpen] = useState(false)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

  // Add local state to maintain key data and track regeneration
  const [currentKeyData, setCurrentKeyData] = useState<KeyResponse | undefined>(keyData)
  const [lastRegeneratedAt, setLastRegeneratedAt] = useState<Date | null>(null)
  const [isRecentlyRegenerated, setIsRecentlyRegenerated] = useState(false)

  // Update local state when keyData prop changes (but don't reset to undefined)
  useEffect(() => {
    if (keyData) {
      setCurrentKeyData(keyData)
    }
  }, [keyData])

  // Reset recent regeneration indicator after 5 seconds
  useEffect(() => {
    if (isRecentlyRegenerated) {
      const timer = setTimeout(() => {
        setIsRecentlyRegenerated(false)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [isRecentlyRegenerated])

  // Use currentKeyData instead of keyData throughout the component
  if (!currentKeyData) {
    return (
      <div className="p-4">
        <Button 
          icon={ArrowLeftIcon} 
          variant="light"
          onClick={onClose}
          className="mb-4"
        >
          Back to Keys
        </Button>
        <Text>Key not found</Text>
      </div>
    );
  }

  const handleKeyUpdate = async (formValues: Record<string, any>) => {
    try {
      if (!accessToken) return;

      const currentKey = formValues.token;
      formValues.key = currentKey;

      // Handle object_permission updates
      if (formValues.vector_stores !== undefined) {
        formValues.object_permission = {
          ...currentKeyData.object_permission,
          vector_stores: formValues.vector_stores || []
        };
        // Remove vector_stores from the top level as it should be in object_permission
        delete formValues.vector_stores;
      }

      if (formValues.mcp_servers_and_groups !== undefined) {
        const { servers, accessGroups } = formValues.mcp_servers_and_groups || { servers: [], accessGroups: [] };
        formValues.object_permission = {
          ...currentKeyData.object_permission,
          mcp_servers: servers || [],
          mcp_access_groups: accessGroups || []
        };
        // Remove mcp_servers_and_groups from the top level as it should be in object_permission
        delete formValues.mcp_servers_and_groups;
      }

      // Convert metadata back to an object if it exists and is a string
      if (formValues.metadata && typeof formValues.metadata === "string") {
        try {
          const parsedMetadata = JSON.parse(formValues.metadata);
          formValues.metadata = {
            ...parsedMetadata,
            ...(formValues.guardrails?.length > 0
              ? { guardrails: formValues.guardrails }
              : {}),
            ...(formValues.logging_settings
              ? { logging: formValues.logging_settings }
              : {}),
            ...(formValues.disabled_callbacks?.length > 0
              ? { 
                  litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks)
                }
              : {}),
          };
        } catch (error) {
          console.error("Error parsing metadata JSON:", error);
          message.error("Invalid metadata JSON");
          return;
        }
      } else {
        formValues.metadata = {
          ...(formValues.metadata || {}),
          ...(formValues.guardrails?.length > 0
            ? { guardrails: formValues.guardrails }
            : {}),
          ...(formValues.logging_settings
            ? { logging: formValues.logging_settings }
            : {}),
          ...(formValues.disabled_callbacks?.length > 0
            ? { 
                litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks)
              }
            : {}),
        };
      }

      delete formValues.logging_settings;

      // Convert budget_duration to API format
      if (formValues.budget_duration) {
        const durationMap: Record<string, string> = {
          daily: "24h",
          weekly: "7d",
          monthly: "30d"
        };
        formValues.budget_duration = durationMap[formValues.budget_duration];
      }

      const newKeyValues = await keyUpdateCall(accessToken, formValues)

      // Update local state
      setCurrentKeyData((prevData) => (prevData ? { ...prevData, ...newKeyValues } : undefined))

      if (onKeyDataUpdate) {
        onKeyDataUpdate(newKeyValues)
      }
      message.success("Key updated successfully");
      setIsEditing(false);
      // Refresh key data here if needed
    } catch (error) {
      message.error("Failed to update key");
      console.error("Error updating key:", error);
    }
  };

  const handleDelete = async () => {
    try {
      if (!accessToken) return
      await keyDeleteCall(accessToken as string, currentKeyData.token || currentKeyData.token_id)
      message.success("Key deleted successfully")
      if (onDelete) {
        onDelete()
      }
      onClose();
    } catch (error) {
      console.error("Error deleting the key:", error);
      message.error("Failed to delete key");
    }
  };

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }))
      }, 2000)
    }
  }

  const handleRegenerateKeyUpdate = (updatedKeyData: Partial<KeyResponse>) => {
    // Update local state immediately with ALL the new data
    setCurrentKeyData((prevData) => {
      if (!prevData) return undefined
      const newData = {
        ...prevData,
        ...updatedKeyData, // This should include the new token (key-id)
        // Update the created_at to show when it was regenerated
        created_at: new Date().toLocaleString(),
      }
      return newData
    })

    // Track regeneration timestamp
    setLastRegeneratedAt(new Date())
    setIsRecentlyRegenerated(true)

    if (onKeyDataUpdate) {
      onKeyDataUpdate({
        ...updatedKeyData,
        created_at: new Date().toLocaleString(),
      })
    }
  }

  // Update the formatTimestamp function to use the desired date format
  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp)
    const dateStr = date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    })
    const timeStr = date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    })
    return `${dateStr} at ${timeStr}`
  }

  return (
    <div className="w-full h-screen p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button 
            icon={ArrowLeftIcon} 
            variant="light"
            onClick={onClose}
            className="mb-4"
          >
            Back to Keys
          </Button>
          <Title>{currentKeyData.key_alias || "API Key"}</Title>

          <div className="flex items-center cursor-pointer mb-2 space-y-6">
            <div>
              <Text className="text-xs text-gray-400 uppercase tracking-wide mt-2">Key ID</Text>
              <Text className="text-gray-500 font-mono text-sm">{currentKeyData.token_id || currentKeyData.token}</Text>
            </div>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["key-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(currentKeyData.token_id || currentKeyData.token, "key-id")}
              className={`ml-2 transition-all duration-200${
                copiedStates["key-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>

          {/* Add timestamp and regeneration indicator */}
          <div className="flex items-center gap-2 flex-wrap">
            <Text className="text-sm text-gray-500">
              {currentKeyData.updated_at && currentKeyData.updated_at !== currentKeyData.created_at
                ? `Updated: ${formatTimestamp(currentKeyData.updated_at)}`
                : `Created: ${formatTimestamp(currentKeyData.created_at)}`}
            </Text>

            {isRecentlyRegenerated && (
              <Badge color="green" size="xs" className="animate-pulse">
                Recently Regenerated
              </Badge>
            )}

            {lastRegeneratedAt && (
              <Badge color="blue" size="xs">
                Regenerated
              </Badge>
            )}
          </div>
        </div>
        {userRole && rolesWithWriteAccess.includes(userRole) && (
          <div className="flex gap-2">
            <Tooltip title={!premiumUser ? "This is a LiteLLM Enterprise feature, and requires a valid key to use." : ""}>
              <span className="inline-block">
                <Button
                  icon={RefreshIcon}
                  variant="secondary"
                  onClick={() => setIsRegenerateModalOpen(true)}
                  className="flex items-center"
                  disabled={!premiumUser}
                >
                  Regenerate Key
                </Button>
              </span>
            </Tooltip>
            <Button
              icon={TrashIcon}
              variant="secondary"
              onClick={() => setIsDeleteModalOpen(true)}
              className="flex items-center"
            >
              Delete Key
            </Button>
          </div>
        )}
      </div>

      {/* Add RegenerateKeyModal */}
      <RegenerateKeyModal
        selectedToken={currentKeyData}
        visible={isRegenerateModalOpen}
        onClose={() => setIsRegenerateModalOpen(false)}
        accessToken={accessToken}
        premiumUser={premiumUser}
        setAccessToken={setAccessToken}
        onKeyUpdate={handleRegenerateKeyUpdate}
      />

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
                      Delete Key
                    </h3>
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">
                        Are you sure you want to delete this key?
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
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          {/* Overview Panel */}
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Spend</Text>
                <div className="mt-2">
                  <Title>${formatNumberWithCommas(currentKeyData.spend, 4)}</Title>
                  <Text>
                    of{" "}
                    {currentKeyData.max_budget !== null
                      ? `$${formatNumberWithCommas(currentKeyData.max_budget)}`
                      : "Unlimited"}
                  </Text>
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {currentKeyData.tpm_limit !== null ? currentKeyData.tpm_limit : "Unlimited"}</Text>
                  <Text>RPM: {currentKeyData.rpm_limit !== null ? currentKeyData.rpm_limit : "Unlimited"}</Text>
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {currentKeyData.models && currentKeyData.models.length > 0 ? (
                    currentKeyData.models.map((model, index) => (
                      <Badge key={index} color="red">
                        {model}
                      </Badge>
                    ))
                  ) : (
                    <Text>No models specified</Text>
                  )}
                </div>
              </Card>

              <Card>
                <ObjectPermissionsView
                  objectPermission={currentKeyData.object_permission}
                  variant="inline"
                  accessToken={accessToken}
                />
              </Card>

              <LoggingSettingsView
                loggingConfigs={extractLoggingSettings(currentKeyData.metadata)}
                disabledCallbacks={
                  Array.isArray(currentKeyData.metadata?.litellm_disabled_callbacks)
                    ? mapInternalToDisplayNames(currentKeyData.metadata.litellm_disabled_callbacks)
                    : []
                }
                variant="card"
              />
            </Grid>
          </TabPanel>

          {/* Settings Panel */}
          <TabPanel>
            <Card className="overflow-y-auto max-h-[65vh]">
              <div className="flex justify-between items-center mb-4">
                <Title>Key Settings</Title>
                {!isEditing && userRole && rolesWithWriteAccess.includes(userRole) && (
                  <Button variant="light" onClick={() => setIsEditing(true)}>
                    Edit Settings
                  </Button>
                )}
              </div>

              {isEditing ? (
                <KeyEditView
                  keyData={currentKeyData}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleKeyUpdate}
                  teams={teams}
                  accessToken={accessToken}
                  userID={userID}
                  userRole={userRole}
                />
              ) : (
                <div className="space-y-4">
                  <div>
                    <Text className="font-medium">Key ID</Text>
                    <Text className="font-mono">{currentKeyData.token_id || currentKeyData.token}</Text>
                  </div>
                  
                  <div>
                    <Text className="font-medium">Key Alias</Text>
                    <Text>{currentKeyData.key_alias || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Secret Key</Text>
                    <Text className="font-mono">{currentKeyData.key_name}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Team ID</Text>
                    <Text>{currentKeyData.team_id || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Organization</Text>
                    <Text>{currentKeyData.organization_id || "Not Set"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Created</Text>
                    <Text>{formatTimestamp(currentKeyData.created_at)}</Text>
                  </div>

                  {lastRegeneratedAt && (
                    <div>
                      <Text className="font-medium">Last Regenerated</Text>
                      <div className="flex items-center gap-2">
                        <Text>{formatTimestamp(lastRegeneratedAt)}</Text>
                        <Badge color="green" size="xs">
                          Recent
                        </Badge>
                      </div>
                    </div>
                  )}

                  <div>
                    <Text className="font-medium">Expires</Text>
                    <Text>{currentKeyData.expires ? formatTimestamp(currentKeyData.expires) : "Never"}</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Spend</Text>
                    <Text>${formatNumberWithCommas(currentKeyData.spend, 4)} USD</Text>
                  </div>

                  <div>
                    <Text className="font-medium">Budget</Text>
                    <Text>
                      {currentKeyData.max_budget !== null
                        ? `$${formatNumberWithCommas(currentKeyData.max_budget, 2)}`
                        : "Unlimited"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Models</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {currentKeyData.models && currentKeyData.models.length > 0 ? (
                        currentKeyData.models.map((model, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-100 rounded text-xs">
                            {model}
                          </span>
                        ))
                      ) : (
                        <Text>No models specified</Text>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Rate Limits</Text>
                    <Text>TPM: {currentKeyData.tpm_limit !== null ? currentKeyData.tpm_limit : "Unlimited"}</Text>
                    <Text>RPM: {currentKeyData.rpm_limit !== null ? currentKeyData.rpm_limit : "Unlimited"}</Text>
                    <Text>
                      Max Parallel Requests:{" "}
                      {currentKeyData.max_parallel_requests !== null
                        ? currentKeyData.max_parallel_requests
                        : "Unlimited"}
                    </Text>
                    <Text>
                      Model TPM Limits:{" "}
                      {currentKeyData.metadata?.model_tpm_limit
                        ? JSON.stringify(currentKeyData.metadata.model_tpm_limit)
                        : "Unlimited"}
                    </Text>
                    <Text>
                      Model RPM Limits:{" "}
                      {currentKeyData.metadata?.model_rpm_limit
                        ? JSON.stringify(currentKeyData.metadata.model_rpm_limit)
                        : "Unlimited"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Metadata</Text>
                    <pre className="bg-gray-100 p-2 rounded text-xs overflow-auto mt-1">
                      {formatMetadataForDisplay(currentKeyData.metadata)}
                    </pre>
                  </div>

                  <ObjectPermissionsView
                    objectPermission={currentKeyData.object_permission}
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                    accessToken={accessToken}
                  />

                  <LoggingSettingsView
                    loggingConfigs={extractLoggingSettings(currentKeyData.metadata)}
                    disabledCallbacks={
                      Array.isArray(currentKeyData.metadata?.litellm_disabled_callbacks)
                        ? mapInternalToDisplayNames(currentKeyData.metadata.litellm_disabled_callbacks)
                        : []
                    }
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                  />
                </div>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
} 