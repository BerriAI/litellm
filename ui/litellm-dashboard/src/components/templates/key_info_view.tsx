import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useTeams from "@/app/(dashboard)/hooks/useTeams";
import { formatNumberWithCommas, copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { mapEmptyStringToNull } from "@/utils/keyUpdateUtils";
import { ArrowLeftIcon, RefreshIcon, TrashIcon } from "@heroicons/react/outline";
import { Badge, Button, Card, Grid, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import { Button as AntdButton, Form, Tag, Tooltip } from "antd";
import { CheckIcon, CopyIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { isProxyAdminRole, isUserTeamAdminForSingleTeam, rolesWithWriteAccess } from "../../utils/roles";
import { mapDisplayToInternalNames, mapInternalToDisplayNames } from "../callback_info_helpers";
import AutoRotationView from "../common_components/AutoRotationView";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import { extractLoggingSettings, formatMetadataForDisplay, stripTagsFromMetadata } from "../key_info_utils";
import { KeyResponse } from "../key_team_helpers/key_list";
import LoggingSettingsView from "../logging_settings_view";
import NotificationManager from "../molecules/notifications_manager";
import { getPolicyInfoWithGuardrails, keyDeleteCall, keyUpdateCall } from "../networking";
import ObjectPermissionsView from "../object_permissions_view";
import { RegenerateKeyModal } from "../organisms/regenerate_key_modal";
import { parseErrorMessage } from "../shared/errorUtils";
import { KeyEditView } from "./key_edit_view";

interface KeyInfoViewProps {
  keyId: string;
  onClose: () => void;
  keyData: KeyResponse | undefined;
  onKeyDataUpdate?: (data: Partial<KeyResponse>) => void;
  onDelete?: () => void;
  teams: any[] | null;
  backButtonText?: string;
}

/**
 * ─────────────────────────────────────────────────────────────────────────
 * @deprecated
 * This component is being DEPRECATED in favor of src/app/(dashboard)/virtual-keys/components/KeyInfoView.tsx
 * Please contribute to the new refactor.
 * ─────────────────────────────────────────────────────────────────────────
 */
export default function KeyInfoView({
  onClose,
  keyData,
  teams,
  onKeyDataUpdate,
  onDelete,
  backButtonText = "Back to Keys",
}: KeyInfoViewProps) {
  const { accessToken, userId: userID, userRole, premiumUser } = useAuthorized();
  const { teams: teamsData } = useTeams();
  const [isEditing, setIsEditing] = useState(false);
  const [form] = Form.useForm();
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");
  const [isRegenerateModalOpen, setIsRegenerateModalOpen] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  // Add local state to maintain key data and track regeneration
  const [currentKeyData, setCurrentKeyData] = useState<KeyResponse | undefined>(keyData);
  const [lastRegeneratedAt, setLastRegeneratedAt] = useState<Date | null>(null);
  const [isRecentlyRegenerated, setIsRecentlyRegenerated] = useState(false);
  const [policyGuardrails, setPolicyGuardrails] = useState<Record<string, string[]>>({});
  const [loadingPolicies, setLoadingPolicies] = useState(false);

  // Update local state when keyData prop changes (but don't reset to undefined)
  useEffect(() => {
    if (keyData) {
      setCurrentKeyData(keyData);
    }
  }, [keyData]);

  // Fetch resolved guardrails for all policies
  useEffect(() => {
    const fetchPolicyGuardrails = async () => {
      const policies = currentKeyData?.metadata?.policies;
      if (!accessToken || !policies || !Array.isArray(policies) || policies.length === 0) {
        return;
      }

      setLoadingPolicies(true);
      const guardrailsMap: Record<string, string[]> = {};

      try {
        await Promise.all(
          policies.map(async (policyName: string) => {
            try {
              const policyInfo = await getPolicyInfoWithGuardrails(accessToken, policyName);
              guardrailsMap[policyName] = policyInfo.resolved_guardrails || [];
            } catch (error) {
              console.error(`Failed to fetch guardrails for policy ${policyName}:`, error);
              guardrailsMap[policyName] = [];
            }
          })
        );
        setPolicyGuardrails(guardrailsMap);
      } catch (error) {
        console.error("Failed to fetch policy guardrails:", error);
      } finally {
        setLoadingPolicies(false);
      }
    };

    fetchPolicyGuardrails();
  }, [accessToken, currentKeyData?.metadata?.policies]);

  // Reset recent regeneration indicator after 5 seconds
  useEffect(() => {
    if (isRecentlyRegenerated) {
      const timer = setTimeout(() => {
        setIsRecentlyRegenerated(false);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [isRecentlyRegenerated]);

  // Use currentKeyData instead of keyData throughout the component
  if (!currentKeyData) {
    return (
      <div className="p-4">
        <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          {backButtonText}
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

      // Guard premium features
      if (!premiumUser) {
        delete formValues.guardrails;
        delete formValues.prompts;
      }

      // Handle max budget empty string
      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);

      // Handle object_permission updates
      if (formValues.vector_stores !== undefined) {
        formValues.object_permission = {
          ...currentKeyData.object_permission,
          vector_stores: formValues.vector_stores || [],
        };
        // Remove vector_stores from the top level as it should be in object_permission
        delete formValues.vector_stores;
      }

      if (formValues.mcp_servers_and_groups !== undefined) {
        const { servers, accessGroups } = formValues.mcp_servers_and_groups || { servers: [], accessGroups: [] };
        formValues.object_permission = {
          ...currentKeyData.object_permission,
          mcp_servers: servers || [],
          mcp_access_groups: accessGroups || [],
        };
        // Remove mcp_servers_and_groups from the top level as it should be in object_permission
        delete formValues.mcp_servers_and_groups;
      }

      // Handle MCP tool permissions
      if (formValues.mcp_tool_permissions !== undefined) {
        const mcpToolPermissions = formValues.mcp_tool_permissions || {};
        if (Object.keys(mcpToolPermissions).length > 0) {
          formValues.object_permission = {
            ...formValues.object_permission,
            mcp_tool_permissions: mcpToolPermissions,
          };
        }
        delete formValues.mcp_tool_permissions;
      }

      // Handle agent permissions
      if (formValues.agents_and_groups !== undefined) {
        const { agents, accessGroups } = formValues.agents_and_groups || { agents: [], accessGroups: [] };
        formValues.object_permission = {
          ...formValues.object_permission,
          agents: agents || [],
          agent_access_groups: accessGroups || [],
        };
        delete formValues.agents_and_groups;
      }

      formValues.max_budget = mapEmptyStringToNull(formValues.max_budget);
      formValues.tpm_limit = mapEmptyStringToNull(formValues.tpm_limit);
      formValues.rpm_limit = mapEmptyStringToNull(formValues.rpm_limit);
      formValues.max_parallel_requests = mapEmptyStringToNull(formValues.max_parallel_requests);

      // Convert metadata back to an object if it exists and is a string
      if (formValues.metadata && typeof formValues.metadata === "string") {
        try {
          const parsedMetadata = JSON.parse(formValues.metadata);
          // Ensure tags are controlled via dedicated field, not in metadata textarea
          if ("tags" in parsedMetadata) {
            delete parsedMetadata["tags"];
          }
          formValues.metadata = {
            ...parsedMetadata,
            ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
            ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
            ...(formValues.logging_settings ? { logging: formValues.logging_settings } : {}),
            ...(formValues.disabled_callbacks?.length > 0
              ? {
                litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks),
              }
              : {}),
          };
        } catch (error) {
          console.error("Error parsing metadata JSON:", error);
          NotificationManager.error("Invalid metadata JSON");
          return;
        }
      } else {
        const baseMetadata = formValues.metadata || {};
        const { tags: _omitTags, ...rest } = baseMetadata;
        formValues.metadata = {
          ...rest,
          ...(Array.isArray(formValues.tags) && formValues.tags.length > 0 ? { tags: formValues.tags } : {}),
          ...(formValues.guardrails?.length > 0 ? { guardrails: formValues.guardrails } : {}),
          ...(formValues.logging_settings ? { logging: formValues.logging_settings } : {}),
          ...(formValues.disabled_callbacks?.length > 0
            ? {
              litellm_disabled_callbacks: mapDisplayToInternalNames(formValues.disabled_callbacks),
            }
            : {}),
        };
      }

      // tags are merged into metadata; do not send as top-level field
      if ("tags" in formValues) {
        delete formValues.tags;
      }
      delete formValues.logging_settings;

      // Convert budget_duration to API format
      if (formValues.budget_duration) {
        const durationMap: Record<string, string> = {
          daily: "24h",
          weekly: "7d",
          monthly: "30d",
        };
        formValues.budget_duration = durationMap[formValues.budget_duration];
      }

      const newKeyValues = await keyUpdateCall(accessToken, formValues);

      // Update local state
      setCurrentKeyData((prevData) => (prevData ? { ...prevData, ...newKeyValues } : undefined));

      if (onKeyDataUpdate) {
        onKeyDataUpdate(newKeyValues);
      }
      NotificationManager.success("Key updated successfully");
      setIsEditing(false);
      // Refresh key data here if needed
    } catch (error) {
      NotificationManager.fromBackend(parseErrorMessage(error));
      console.error("Error updating key:", error);
    }
  };

  const handleDelete = async () => {
    try {
      setDeleteLoading(true);
      if (!accessToken) return;
      await keyDeleteCall(accessToken as string, currentKeyData.token || currentKeyData.token_id);
      NotificationManager.success("Key deleted successfully");
      if (onDelete) {
        onDelete();
      }
      onClose();
    } catch (error) {
      console.error("Error deleting the key:", error);
      NotificationManager.fromBackend(error);
    } finally {
      setDeleteLoading(false);
      setIsDeleteModalOpen(false);
      setDeleteConfirmInput("");
    }
  };

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const handleRegenerateKeyUpdate = (updatedKeyData: Partial<KeyResponse>) => {
    // Update local state immediately with ALL the new data
    setCurrentKeyData((prevData) => {
      if (!prevData) return undefined;
      const newData = {
        ...prevData,
        ...updatedKeyData, // This should include the new token (key-id)
        // Update the created_at to show when it was regenerated
        created_at: new Date().toLocaleString(),
      };
      return newData;
    });

    // Track regeneration timestamp
    setLastRegeneratedAt(new Date());
    setIsRecentlyRegenerated(true);

    if (onKeyDataUpdate) {
      onKeyDataUpdate({
        ...updatedKeyData,
        created_at: new Date().toLocaleString(),
      });
    }
  };

  // Update the formatTimestamp function to use the desired date format
  const formatTimestamp = (timestamp: string | Date) => {
    const date = new Date(timestamp);
    const dateStr = date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
    const timeStr = date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
    return `${dateStr} at ${timeStr}`;
  };

  const canModifyKey =
    isProxyAdminRole(userRole || "") ||
    (teamsData &&
      isUserTeamAdminForSingleTeam(
        teamsData?.filter((team) => team.team_id === currentKeyData.team_id)[0]?.members_with_roles,
        userID || "",
      )) ||
    (userID === currentKeyData.user_id && userRole !== "Internal Viewer");

  return (
    <div className="w-full h-screen p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
            {backButtonText}
          </Button>
          <Title>{currentKeyData.key_alias || "Virtual Key"}</Title>

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
              className={`ml-2 transition-all duration-200${copiedStates["key-id"]
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
        {canModifyKey && (
          <div className="flex gap-2">
            <Tooltip
              title={!premiumUser ? "This is a LiteLLM Enterprise feature, and requires a valid key to use." : ""}
            >
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
              className="flex items-center text-red-500 border-red-500 hover:text-red-700"
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
        onKeyUpdate={handleRegenerateKeyUpdate}
      />

      {/* Delete Confirmation Modal */}
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Key"
        alertMessage="This action is irreversible and will immediately revoke access for any applications using this key."
        message="Are you sure you want to delete this Virtual Key?"
        resourceInformationTitle="Key Information"
        resourceInformation={[
          {
            label: "Key Alias",
            value: currentKeyData?.key_alias || "-",
          },
          {
            label: "Key ID",
            value: currentKeyData?.token_id || currentKeyData?.token || "-",
            code: true,
          },
          {
            label: "Team ID",
            value: currentKeyData?.team_id || "-",
            code: true,
          },
          {
            label: "Spend",
            value: currentKeyData?.spend ? `$${formatNumberWithCommas(currentKeyData.spend, 4)}` : "$0.0000",
          },
        ]}
        onCancel={() => {
          setIsDeleteModalOpen(false);
          setDeleteConfirmInput("");
        }}
        onOk={handleDelete}
        confirmLoading={deleteLoading}
        requiredConfirmation={currentKeyData?.key_alias}
      />

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

              <Card>
                <Text className="font-medium mb-3">Guardrails</Text>
                {Array.isArray(currentKeyData.metadata?.guardrails) && currentKeyData.metadata.guardrails.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {currentKeyData.metadata.guardrails.map((guardrail: string, index: number) => (
                      <Badge key={index} color="blue">
                        {guardrail}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <Text className="text-gray-500">No guardrails configured</Text>
                )}
                {typeof currentKeyData.metadata?.disable_global_guardrails === "boolean" &&
                  currentKeyData.metadata.disable_global_guardrails === true && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <Badge color="yellow">Global Guardrails Disabled</Badge>
                    </div>
                  )}
              </Card>

              <Card>
                <Text className="font-medium mb-3">Policies</Text>
                {Array.isArray(currentKeyData.metadata?.policies) && currentKeyData.metadata.policies.length > 0 ? (
                  <div className="space-y-4">
                    {currentKeyData.metadata.policies.map((policy: string, index: number) => (
                      <div key={index} className="space-y-2">
                        <div className="flex items-center gap-2">
                          <Badge color="purple">{policy}</Badge>
                          {loadingPolicies && <Text className="text-xs text-gray-400">Loading guardrails...</Text>}
                        </div>
                        {!loadingPolicies && policyGuardrails[policy] && policyGuardrails[policy].length > 0 && (
                          <div className="ml-4 pl-3 border-l-2 border-gray-200">
                            <Text className="text-xs text-gray-500 mb-1">Resolved Guardrails:</Text>
                            <div className="flex flex-wrap gap-1">
                              {policyGuardrails[policy].map((guardrail: string, gIndex: number) => (
                                <Badge key={gIndex} color="blue" size="xs">
                                  {guardrail}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <Text className="text-gray-500">No policies configured</Text>
                )}
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

              <AutoRotationView
                autoRotate={currentKeyData.auto_rotate}
                rotationInterval={currentKeyData.rotation_interval}
                lastRotationAt={currentKeyData.last_rotation_at}
                keyRotationAt={currentKeyData.key_rotation_at}
                nextRotationAt={currentKeyData.next_rotation_at}
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
                  <Button onClick={() => setIsEditing(true)}>Edit Settings</Button>
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
                  premiumUser={premiumUser}
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

                  <AutoRotationView
                    autoRotate={currentKeyData.auto_rotate}
                    rotationInterval={currentKeyData.rotation_interval}
                    lastRotationAt={currentKeyData.last_rotation_at}
                    keyRotationAt={currentKeyData.key_rotation_at}
                    nextRotationAt={currentKeyData.next_rotation_at}
                    variant="inline"
                    className="pt-4 border-t border-gray-200"
                  />

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
                    <Text className="font-medium">Tags</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Array.isArray(currentKeyData.metadata?.tags) && currentKeyData.metadata.tags.length > 0
                        ? currentKeyData.metadata.tags.map((tag, index) => (
                          <span key={index} className="px-2 mr-2 py-1 bg-blue-100 rounded text-xs">
                            {tag}
                          </span>
                        ))
                        : "No tags specified"}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Prompts</Text>
                    <Text>
                      {Array.isArray(currentKeyData.metadata?.prompts) && currentKeyData.metadata.prompts.length > 0
                        ? currentKeyData.metadata.prompts.map((prompt, index) => (
                          <span key={index} className="px-2 mr-2 py-1 bg-blue-100 rounded text-xs">
                            {prompt}
                          </span>
                        ))
                        : "No prompts specified"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Allowed Routes</Text>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Array.isArray(currentKeyData.allowed_routes) && currentKeyData.allowed_routes.length > 0 ? (
                        currentKeyData.allowed_routes.map((route, index) => (
                          <span key={index} className="px-2 py-1 bg-blue-100 rounded text-xs">
                            {route}
                          </span>
                        ))
                      ) : (
                        <Tag color="green">All routes allowed</Tag>
                      )}
                    </div>
                  </div>

                  <div>
                    <Text className="font-medium">Allowed Pass Through Routes</Text>
                    <Text>
                      {Array.isArray(currentKeyData.metadata?.allowed_passthrough_routes) &&
                        currentKeyData.metadata.allowed_passthrough_routes.length > 0
                        ? currentKeyData.metadata.allowed_passthrough_routes.map((route, index) => (
                          <span key={index} className="px-2 mr-2 py-1 bg-blue-100 rounded text-xs">
                            {route}
                          </span>
                        ))
                        : "No pass through routes specified"}
                    </Text>
                  </div>

                  <div>
                    <Text className="font-medium">Disable Global Guardrails</Text>
                    <Text>
                      {currentKeyData.metadata?.disable_global_guardrails === true ? (
                        <Badge color="yellow">Enabled - Global guardrails bypassed</Badge>
                      ) : (
                        <Badge color="green">Disabled - Global guardrails active</Badge>
                      )}
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
                      {formatMetadataForDisplay(stripTagsFromMetadata(currentKeyData.metadata))}
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
