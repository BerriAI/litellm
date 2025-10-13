"use client";
import React, { useState, useEffect, useCallback } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { Text, Title, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button as Button2, Modal, Form, Input, Select, Radio } from "antd";
import NumericalInput from "../shared/numerical_input";
import { getModelDisplayName } from "../key_team_helpers/fetch_available_models_team_key";
import SchemaFormFields from "../common_components/check_openapi_schema";
import {
  keyCreateCall,
  modelAvailableCall,
  getGuardrailsList,
  proxyBaseUrl,
  getPossibleUserRoles,
  userFilterUICall,
  keyCreateServiceAccountCall,
  fetchMCPAccessGroups,
  getPromptsList,
} from "../networking";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";
import { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "../common_components/team_dropdown";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Tooltip } from "antd";
import PremiumLoggingSettings from "../common_components/PremiumLoggingSettings";
import Createuser from "../create_user_button";
import debounce from "lodash/debounce";
import { rolesWithWriteAccess } from "../../utils/roles";
import BudgetDurationDropdown from "../common_components/budget_duration_dropdown";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { mapDisplayToInternalNames } from "../callback_info_helpers";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import ModelAliasManager from "../common_components/ModelAliasManager";
import NotificationsManager from "../molecules/notifications_manager";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  team: Team | null;
  userRole: string | null;
  accessToken: string;
  data: any[] | null;
  teams: Team[] | null;
  addKey: (data: any) => void;
  premiumUser?: boolean;
}

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}

const getPredefinedTags = (data: any[] | null) => {
  let allTags = [];

  console.log("data:", JSON.stringify(data));

  if (data) {
    for (let key of data) {
      if (key["metadata"] && key["metadata"]["tags"]) {
        allTags.push(...key["metadata"]["tags"]);
      }
    }
  }

  // Deduplicate using Set
  const uniqueTags = Array.from(new Set(allTags)).map((tag) => ({
    value: tag,
    label: tag,
  }));

  console.log("uniqueTags:", uniqueTags);
  return uniqueTags;
};

export const fetchTeamModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  teamID: string | null,
): Promise<string[]> => {
  try {
    if (userID === null || userRole === null) {
      return [];
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole, true, teamID, true);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      console.log("available_model_names:", available_model_names);
      return available_model_names;
    }
    return [];
  } catch (error) {
    console.error("Error fetching user models:", error);
    return [];
  }
};

export const fetchUserModels = async (
  userID: string,
  userRole: string,
  accessToken: string,
  setUserModels: (models: string[]) => void,
) => {
  try {
    if (userID === null || userRole === null) {
      return;
    }

    if (accessToken !== null) {
      const model_available = await modelAvailableCall(accessToken, userID, userRole);
      let available_model_names = model_available["data"].map((element: { id: string }) => element.id);
      console.log("available_model_names:", available_model_names);
      setUserModels(available_model_names);
    }
  } catch (error) {
    console.error("Error fetching user models:", error);
  }
};

/**
 * ─────────────────────────────────────────────────────────────────────────
 * @deprecated
 * This component is being DEPRECATED in favor of src/app/(dashboard)/virtual-keys/components/CreateKey.tsx
 * Please contribute to the new refactor.
 * ─────────────────────────────────────────────────────────────────────────
 */
const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  team,
  teams,
  userRole,
  accessToken,
  data,
  addKey,
  premiumUser = false,
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);
  const [softBudget, setSoftBudget] = useState(null);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [keyOwner, setKeyOwner] = useState("you");
  const [predefinedTags, setPredefinedTags] = useState(getPredefinedTags(data));
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [selectedCreateKeyTeam, setSelectedCreateKeyTeam] = useState<Team | null>(team);
  const [isCreateUserModalVisible, setIsCreateUserModalVisible] = useState(false);
  const [newlyCreatedUserId, setNewlyCreatedUserId] = useState<string | null>(null);
  const [possibleUIRoles, setPossibleUIRoles] = useState<Record<string, Record<string, string>>>({});
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [userSearchLoading, setUserSearchLoading] = useState<boolean>(false);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>([]);
  const [keyType, setKeyType] = useState<string>("default");
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(false);
  const [rotationInterval, setRotationInterval] = useState<string>("30d");

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("default");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    setSelectedCreateKeyTeam(null);
    form.resetFields();
    setLoggingSettings([]);
    setDisabledCallbacks([]);
    setKeyType("default");
    setModelAliases({});
    setAutoRotationEnabled(false);
    setRotationInterval("30d");
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
  }, [accessToken, userID, userRole]);

  const fetchMcpAccessGroups = async () => {
    try {
      if (accessToken == null) {
        return;
      }
      const groups = await fetchMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

  useEffect(() => {
    fetchMcpAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchPrompts = async () => {
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(response.prompts.map((prompt) => prompt.prompt_id));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    fetchGuardrails();
    fetchPrompts();
  }, [accessToken]);

  // Fetch possible user roles when component mounts
  useEffect(() => {
    const fetchPossibleRoles = async () => {
      try {
        if (accessToken) {
          // Check if roles are cached in session storage
          const cachedRoles = sessionStorage.getItem("possibleUserRoles");
          if (cachedRoles) {
            setPossibleUIRoles(JSON.parse(cachedRoles));
          } else {
            const availableUserRoles = await getPossibleUserRoles(accessToken);
            sessionStorage.setItem("possibleUserRoles", JSON.stringify(availableUserRoles));
            setPossibleUIRoles(availableUserRoles);
          }
        }
      } catch (error) {
        console.error("Error fetching possible user roles:", error);
      }
    };

    fetchPossibleRoles();
  }, [accessToken]);

  // Check if team selection is required
  const isTeamSelectionRequired = modelsToPick.includes("no-default-models");
  const isFormDisabled = isTeamSelectionRequired && !selectedCreateKeyTeam;

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      const newKeyAlias = formValues?.key_alias ?? "";
      const newKeyTeamId = formValues?.team_id ?? null;

      const existingKeyAliases = data?.filter((k) => k.team_id === newKeyTeamId).map((k) => k.key_alias) ?? [];

      if (existingKeyAliases.includes(newKeyAlias)) {
        throw new Error(
          `Key alias ${newKeyAlias} already exists for team with ID ${newKeyTeamId}, please provide another key alias`,
        );
      }

      NotificationsManager.info("Making API Call");
      setIsModalVisible(true);

      if (keyOwner === "you") {
        formValues.user_id = userID;
      }

      // Handle metadata for all key types
      let metadata: Record<string, any> = {};
      try {
        metadata = JSON.parse(formValues.metadata || "{}");
      } catch (error) {
        console.error("Error parsing metadata:", error);
      }

      // If it's a service account, add the service_account_id to the metadata
      if (keyOwner === "service_account") {
        metadata["service_account_id"] = formValues.key_alias;
      }

      // Add logging settings to the metadata
      if (loggingSettings.length > 0) {
        metadata = {
          ...metadata,
          logging: loggingSettings.filter((config) => config.callback_name),
        };
      }

      // Add disabled callbacks to the metadata
      if (disabledCallbacks.length > 0) {
        // Map display names to internal callback values
        const mappedDisabledCallbacks = mapDisplayToInternalNames(disabledCallbacks);
        metadata = {
          ...metadata,
          litellm_disabled_callbacks: mappedDisabledCallbacks,
        };
      }

      // Add auto-rotation settings as top-level fields
      if (autoRotationEnabled) {
        formValues.auto_rotate = true;
        formValues.rotation_interval = rotationInterval;
      }

      // Handle duration field for key expiry
      if (formValues.duration) {
        formValues.duration = formValues.duration;
      }

      // Update the formValues with the final metadata
      formValues.metadata = JSON.stringify(metadata);

      // Transform allowed_vector_store_ids and allowed_mcp_servers_and_groups into object_permission format
      if (formValues.allowed_vector_store_ids && formValues.allowed_vector_store_ids.length > 0) {
        formValues.object_permission = {
          vector_stores: formValues.allowed_vector_store_ids,
        };
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_vector_store_ids;
      }

      // Transform allowed_mcp_servers_and_groups into object_permission format
      if (
        formValues.allowed_mcp_servers_and_groups &&
        (formValues.allowed_mcp_servers_and_groups.servers?.length > 0 ||
          formValues.allowed_mcp_servers_and_groups.accessGroups?.length > 0)
      ) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        const { servers, accessGroups } = formValues.allowed_mcp_servers_and_groups;
        if (servers && servers.length > 0) {
          formValues.object_permission.mcp_servers = servers;
        }
        if (accessGroups && accessGroups.length > 0) {
          formValues.object_permission.mcp_access_groups = accessGroups;
        }
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_mcp_servers_and_groups;
      }

      // Add MCP tool permissions to object_permission
      const mcpToolPermissions = formValues.mcp_tool_permissions || {};
      if (Object.keys(mcpToolPermissions).length > 0) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        formValues.object_permission.mcp_tool_permissions = mcpToolPermissions;
      }
      delete formValues.mcp_tool_permissions;

      // Transform allowed_mcp_access_groups into object_permission format
      if (formValues.allowed_mcp_access_groups && formValues.allowed_mcp_access_groups.length > 0) {
        if (!formValues.object_permission) {
          formValues.object_permission = {};
        }
        formValues.object_permission.mcp_access_groups = formValues.allowed_mcp_access_groups;
        // Remove the original field as it's now part of object_permission
        delete formValues.allowed_mcp_access_groups;
      }

      // Add model_aliases if any are defined
      if (Object.keys(modelAliases).length > 0) {
        formValues.aliases = JSON.stringify(modelAliases);
      }

      let response;
      if (keyOwner === "service_account") {
        response = await keyCreateServiceAccountCall(accessToken, formValues);
      } else {
        response = await keyCreateCall(accessToken, userID, formValues);
      }

      console.log("key create Response:", response);

      // Add the data to the state in the parent component
      // Also directly update the keys list in AllKeysTable without an API call
      addKey(response);

      setApiKey(response["key"]);
      setSoftBudget(response["soft_budget"]);
      NotificationsManager.success("API Key Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.log("error in create key:", error);
      NotificationsManager.fromBackend(`Error creating the key: ${error}`);
    }
  };

  const handleCopy = () => {
    NotificationsManager.success("API Key copied to clipboard");
  };

  useEffect(() => {
    if (userID && userRole && accessToken) {
      fetchTeamModels(userID, userRole, accessToken, selectedCreateKeyTeam?.team_id ?? null).then((models) => {
        let allModels = Array.from(new Set([...(selectedCreateKeyTeam?.models ?? []), ...models]));
        setModelsToPick(allModels);
      });
    }
    form.setFieldValue("models", []);
  }, [selectedCreateKeyTeam, accessToken, userID, userRole]);

  // Add a callback function to handle user creation
  const handleUserCreated = (userId: string) => {
    setNewlyCreatedUserId(userId);
    form.setFieldsValue({ user_id: userId });
    setIsCreateUserModalVisible(false);
  };

  const fetchUsers = async (searchText: string): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }

    setUserSearchLoading(true);
    try {
      const params = new URLSearchParams();
      params.append("user_email", searchText); // Always search by email
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);

      const data: User[] = response;
      const options: UserOption[] = data.map((user) => ({
        label: `${user.user_email} (${user.user_id})`,
        value: user.user_id,
        user,
      }));

      setUserOptions(options);
    } catch (error) {
      console.error("Error fetching users:", error);
      NotificationsManager.fromBackend("Failed to search for users");
    } finally {
      setUserSearchLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce((text: string) => fetchUsers(text), 300),
    [accessToken],
  );

  const handleUserSearch = (value: string): void => {
    debouncedSearch(value);
  };

  const handleUserSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_id: selectedUser.user_id,
    });
  };

  return (
    <div>
      {userRole && rolesWithWriteAccess.includes(userRole) && (
        <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
          + Create New Key
        </Button>
      )}
      <Modal
        // title="Create Key"
        visible={isModalVisible}
        width={1000}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
          {/* Section 1: Key Ownership */}
          <div className="mb-8">
            <Title className="mb-4">Key Ownership</Title>
            <Form.Item
              label={
                <span>
                  Owned By{" "}
                  <Tooltip title="Select who will own this API key">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              className="mb-4"
            >
              <Radio.Group onChange={(e) => setKeyOwner(e.target.value)} value={keyOwner}>
                <Radio value="you">You</Radio>
                <Radio value="service_account">Service Account</Radio>
                {userRole === "Admin" && <Radio value="another_user">Another User</Radio>}
              </Radio.Group>
            </Form.Item>

            {keyOwner === "another_user" && (
              <Form.Item
                label={
                  <span>
                    User ID{" "}
                    <Tooltip title="The user who will own this key and be responsible for its usage">
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="user_id"
                className="mt-4"
                rules={[
                  {
                    required: keyOwner === "another_user",
                    message: `Please input the user ID of the user you are assigning the key to`,
                  },
                ]}
              >
                <div>
                  <div style={{ display: "flex", marginBottom: "8px" }}>
                    <Select
                      showSearch
                      placeholder="Type email to search for users"
                      filterOption={false}
                      onSearch={handleUserSearch}
                      onSelect={(value, option) => handleUserSelect(value, option as UserOption)}
                      options={userOptions}
                      loading={userSearchLoading}
                      allowClear
                      style={{ width: "100%" }}
                      notFoundContent={userSearchLoading ? "Searching..." : "No users found"}
                    />
                    <Button2 onClick={() => setIsCreateUserModalVisible(true)} style={{ marginLeft: "8px" }}>
                      Create User
                    </Button2>
                  </div>
                  <div className="text-xs text-gray-500">Search by email to find users</div>
                </div>
              </Form.Item>
            )}
            <Form.Item
              label={
                <span>
                  Team{" "}
                  <Tooltip title="The team this key belongs to, which determines available models and budget limits">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              name="team_id"
              initialValue={team ? team.team_id : null}
              className="mt-4"
              rules={[
                {
                  required: keyOwner === "service_account",
                  message: "Please select a team for the service account",
                },
              ]}
              help={keyOwner === "service_account" ? "required" : ""}
            >
              <TeamDropdown
                teams={teams}
                onChange={(teamId) => {
                  const selectedTeam = teams?.find((t) => t.team_id === teamId) || null;
                  setSelectedCreateKeyTeam(selectedTeam);
                }}
              />
            </Form.Item>
          </div>

          {/* Show message when team selection is required */}
          {isFormDisabled && (
            <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-md">
              <Text className="text-blue-800 text-sm">
                Please select a team to continue configuring your API key. If you do not see any teams, please contact
                your Proxy Admin to either provide you with access to models or to add you to a team.
              </Text>
            </div>
          )}

          {/* Section 2: Key Details */}
          {!isFormDisabled && (
            <div className="mb-8">
              <Title className="mb-4">Key Details</Title>
              <Form.Item
                label={
                  <span>
                    {keyOwner === "you" || keyOwner === "another_user" ? "Key Name" : "Service Account ID"}{" "}
                    <Tooltip
                      title={
                        keyOwner === "you" || keyOwner === "another_user"
                          ? "A descriptive name to identify this key"
                          : "Unique identifier for this service account"
                      }
                    >
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="key_alias"
                rules={[
                  {
                    required: true,
                    message: `Please input a ${keyOwner === "you" ? "key name" : "service account ID"}`,
                  },
                ]}
                help="required"
              >
                <TextInput placeholder="" />
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    Models{" "}
                    <Tooltip title="Select which models this key can access. Choose 'All Team Models' to grant access to all models available to the team">
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="models"
                rules={
                  keyType === "management" || keyType === "read_only"
                    ? []
                    : [{ required: true, message: "Please select a model" }]
                }
                help={
                  keyType === "management" || keyType === "read_only"
                    ? "Models field is disabled for this key type"
                    : "required"
                }
                className="mt-4"
              >
                <Select
                  mode="multiple"
                  placeholder="Select models"
                  style={{ width: "100%" }}
                  disabled={keyType === "management" || keyType === "read_only"}
                  onChange={(values) => {
                    if (values.includes("all-team-models")) {
                      form.setFieldsValue({ models: ["all-team-models"] });
                    }
                  }}
                >
                  <Option key="all-team-models" value="all-team-models">
                    All Team Models
                  </Option>
                  {modelsToPick.map((model: string) => (
                    <Option key={model} value={model}>
                      {getModelDisplayName(model)}
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    Key Type{" "}
                    <Tooltip title="Select the type of key to determine what routes and operations this key can access">
                      <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                    </Tooltip>
                  </span>
                }
                name="key_type"
                initialValue="default"
                className="mt-4"
              >
                <Select
                  defaultValue="default"
                  placeholder="Select key type"
                  style={{ width: "100%" }}
                  optionLabelProp="label"
                  onChange={(value) => {
                    setKeyType(value);
                    // Clear models field and disable if management or read_only
                    if (value === "management" || value === "read_only") {
                      form.setFieldsValue({ models: [] });
                    }
                  }}
                >
                  <Option value="default" label="Default">
                    <div style={{ padding: "4px 0" }}>
                      <div style={{ fontWeight: 500 }}>Default</div>
                      <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                        Can call LLM API + Management routes
                      </div>
                    </div>
                  </Option>
                  <Option value="llm_api" label="LLM API">
                    <div style={{ padding: "4px 0" }}>
                      <div style={{ fontWeight: 500 }}>LLM API</div>
                      <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                        Can call only LLM API routes (chat/completions, embeddings, etc.)
                      </div>
                    </div>
                  </Option>
                  <Option value="management" label="Management">
                    <div style={{ padding: "4px 0" }}>
                      <div style={{ fontWeight: 500 }}>Management</div>
                      <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                        Can call only management routes (user/team/key management)
                      </div>
                    </div>
                  </Option>
                </Select>
              </Form.Item>
            </div>
          )}

          {/* Section 3: Optional Settings */}
          {!isFormDisabled && (
            <div className="mb-8">
              <Accordion className="mt-4 mb-4">
                <AccordionHeader>
                  <Title className="m-0">Optional Settings</Title>
                </AccordionHeader>
                <AccordionBody>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        Max Budget (USD){" "}
                        <Tooltip title="Maximum amount in USD this key can spend. When reached, the key will be blocked from making further requests">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="max_budget"
                    help={`Budget cannot exceed team max budget: $${team?.max_budget !== null && team?.max_budget !== undefined ? team?.max_budget : "unlimited"}`}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.max_budget !== null && value > team.max_budget) {
                            throw new Error(
                              `Budget cannot exceed team max budget: $${formatNumberWithCommas(team.max_budget, 4)}`,
                            );
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={0.01} precision={2} width={200} />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        Reset Budget{" "}
                        <Tooltip title="How often the budget should reset. For example, setting 'daily' will reset the budget every 24 hours">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="budget_duration"
                    help={`Team Reset Budget: ${team?.budget_duration !== null && team?.budget_duration !== undefined ? team?.budget_duration : "None"}`}
                  >
                    <BudgetDurationDropdown onChange={(value) => form.setFieldValue("budget_duration", value)} />
                  </Form.Item>
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        Tokens per minute Limit (TPM){" "}
                        <Tooltip title="Maximum number of tokens this key can process per minute. Helps control usage and costs">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="tpm_limit"
                    help={`TPM cannot exceed team TPM limit: ${team?.tpm_limit !== null && team?.tpm_limit !== undefined ? team?.tpm_limit : "unlimited"}`}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.tpm_limit !== null && value > team.tpm_limit) {
                            throw new Error(`TPM limit cannot exceed team TPM limit: ${team.tpm_limit}`);
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <RateLimitTypeFormItem
                    type="tpm"
                    name="tpm_limit_type"
                    className="mt-4"
                    initialValue={null}
                    form={form}
                    showDetailedDescriptions={true}
                  />
                  <Form.Item
                    className="mt-4"
                    label={
                      <span>
                        Requests per minute Limit (RPM){" "}
                        <Tooltip title="Maximum number of API requests this key can make per minute. Helps prevent abuse and manage load">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="rpm_limit"
                    help={`RPM cannot exceed team RPM limit: ${team?.rpm_limit !== null && team?.rpm_limit !== undefined ? team?.rpm_limit : "unlimited"}`}
                    rules={[
                      {
                        validator: async (_, value) => {
                          if (value && team && team.rpm_limit !== null && value > team.rpm_limit) {
                            throw new Error(`RPM limit cannot exceed team RPM limit: ${team.rpm_limit}`);
                          }
                        },
                      },
                    ]}
                  >
                    <NumericalInput step={1} width={400} />
                  </Form.Item>
                  <RateLimitTypeFormItem
                    type="rpm"
                    name="rpm_limit_type"
                    className="mt-4"
                    initialValue={null}
                    form={form}
                    showDetailedDescriptions={true}
                  />
                  <Form.Item
                    label={
                      <span>
                        Guardrails{" "}
                        <Tooltip title="Apply safety guardrails to this key to filter content or enforce policies">
                          <a
                            href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="guardrails"
                    className="mt-4"
                    help={
                      premiumUser
                        ? "Select existing guardrails or enter new ones"
                        : "Premium feature - Upgrade to set guardrails by key"
                    }
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      disabled={!premiumUser}
                      placeholder={
                        !premiumUser
                          ? "Premium feature - Upgrade to set guardrails by key"
                          : "Select or enter guardrails"
                      }
                      options={guardrailsList.map((name) => ({ value: name, label: name }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        Prompts{" "}
                        <Tooltip title="Allow this key to use specific prompt templates">
                          <a
                            href="https://docs.litellm.ai/docs/proxy/prompt_management"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="prompts"
                    className="mt-4"
                    help={
                      premiumUser
                        ? "Select existing prompts or enter new ones"
                        : "Premium feature - Upgrade to set prompts by key"
                    }
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      disabled={!premiumUser}
                      placeholder={
                        !premiumUser ? "Premium feature - Upgrade to set prompts by key" : "Select or enter prompts"
                      }
                      options={promptsList.map((name) => ({ value: name, label: name }))}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        Allowed Pass Through Routes{" "}
                        <Tooltip title="Allow this key to use specific pass through routes">
                          <a
                            href="https://docs.litellm.ai/docs/proxy/pass_through"
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                          >
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </a>
                        </Tooltip>
                      </span>
                    }
                    name="allowed_passthrough_routes"
                    className="mt-4"
                    help={
                      premiumUser
                        ? "Select existing pass through routes or enter new ones"
                        : "Premium feature - Upgrade to set pass through routes by key"
                    }
                  >
                    <PassThroughRoutesSelector
                      onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
                      value={form.getFieldValue("allowed_passthrough_routes")}
                      accessToken={accessToken}
                      placeholder={
                        !premiumUser ? "Premium feature - Upgrade to set pass through routes by key" : "Select or enter pass through routes"
                      }
                      disabled={!premiumUser}
                      teamId={selectedCreateKeyTeam ? selectedCreateKeyTeam.team_id : null}
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        Allowed Vector Stores{" "}
                        <Tooltip title="Select which vector stores this key can access. If none selected, the key will have access to all available vector stores">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="allowed_vector_store_ids"
                    className="mt-4"
                    help="Select vector stores this key can access. Leave empty for access to all vector stores"
                  >
                    <VectorStoreSelector
                      onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
                      value={form.getFieldValue("allowed_vector_store_ids")}
                      accessToken={accessToken}
                      placeholder="Select vector stores (optional)"
                    />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        Metadata{" "}
                        <Tooltip title="JSON object with additional information about this key. Used for tracking or custom logic">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="metadata"
                    className="mt-4"
                  >
                    <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
                  </Form.Item>
                  <Form.Item
                    label={
                      <span>
                        Tags{" "}
                        <Tooltip title="Tags for tracking spend and/or doing tag-based routing. Used for analytics and filtering">
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    name="tags"
                    className="mt-4"
                    help={`Tags for tracking spend and/or doing tag-based routing.`}
                  >
                    <Select
                      mode="tags"
                      style={{ width: "100%" }}
                      placeholder="Enter tags"
                      tokenSeparators={[","]}
                      options={predefinedTags}
                    />
                  </Form.Item>
                  <Accordion
                    className="mt-4 mb-4"
                    onClick={() => {
                      if (!mcpAccessGroupsLoaded) {
                        fetchMcpAccessGroups();
                        setMcpAccessGroupsLoaded(true);
                      }
                    }}
                  >
                    <AccordionHeader>
                      <b>MCP Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <Form.Item
                        label={
                          <span>
                            Allowed MCP Servers{" "}
                            <Tooltip title="Select which MCP servers or access groups this key can access">
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </Tooltip>
                          </span>
                        }
                        name="allowed_mcp_servers_and_groups"
                        help="Select MCP servers or access groups this key can access"
                      >
                        <MCPServerSelector
                          onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
                          value={form.getFieldValue("allowed_mcp_servers_and_groups")}
                          accessToken={accessToken}
                          placeholder="Select MCP servers or access groups (optional)"
                        />
                      </Form.Item>

                      {/* Hidden field to register mcp_tool_permissions with the form */}
                      <Form.Item name="mcp_tool_permissions" initialValue={{}} hidden>
                        <Input type="hidden" />
                      </Form.Item>

                      <Form.Item
                        noStyle
                        shouldUpdate={(prevValues, currentValues) =>
                          prevValues.allowed_mcp_servers_and_groups !== currentValues.allowed_mcp_servers_and_groups ||
                          prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
                        }
                      >
                        {() => (
                          <div className="mt-6">
                            <MCPToolPermissions
                              accessToken={accessToken}
                              selectedServers={form.getFieldValue("allowed_mcp_servers_and_groups")?.servers || []}
                              toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
                              onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
                            />
                          </div>
                        )}
                      </Form.Item>
                    </AccordionBody>
                  </Accordion>

                  {premiumUser ? (
                    <Accordion className="mt-4 mb-4">
                      <AccordionHeader>
                        <b>Logging Settings</b>
                      </AccordionHeader>
                      <AccordionBody>
                        <div className="mt-4">
                          <PremiumLoggingSettings
                            value={loggingSettings}
                            onChange={setLoggingSettings}
                            premiumUser={true}
                            disabledCallbacks={disabledCallbacks}
                            onDisabledCallbacksChange={setDisabledCallbacks}
                          />
                        </div>
                      </AccordionBody>
                    </Accordion>
                  ) : (
                    <Tooltip
                      title={
                        <span>
                          Key-level logging settings is an enterprise feature, get in touch -
                          <a href="https://www.litellm.ai/enterprise" target="_blank">
                            https://www.litellm.ai/enterprise
                          </a>
                        </span>
                      }
                      placement="top"
                    >
                      <div style={{ position: "relative" }}>
                        <div style={{ opacity: 0.5 }}>
                          <Accordion className="mt-4 mb-4">
                            <AccordionHeader>
                              <b>Logging Settings</b>
                            </AccordionHeader>
                            <AccordionBody>
                              <div className="mt-4">
                                <PremiumLoggingSettings
                                  value={loggingSettings}
                                  onChange={setLoggingSettings}
                                  premiumUser={false}
                                  disabledCallbacks={disabledCallbacks}
                                  onDisabledCallbacksChange={setDisabledCallbacks}
                                />
                              </div>
                            </AccordionBody>
                          </Accordion>
                        </div>
                        <div style={{ position: "absolute", inset: 0, cursor: "not-allowed" }} />
                      </div>
                    </Tooltip>
                  )}

                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>Model Aliases</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <Text className="text-sm text-gray-600 mb-4">
                          Create custom aliases for models that can be used in API calls. This allows you to create
                          shortcuts for specific models.
                        </Text>
                        <ModelAliasManager
                          accessToken={accessToken}
                          initialModelAliases={modelAliases}
                          onAliasUpdate={setModelAliases}
                          showExampleConfig={false}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>

                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>Key Lifecycle</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <KeyLifecycleSettings
                          form={form}
                          autoRotationEnabled={autoRotationEnabled}
                          onAutoRotationChange={setAutoRotationEnabled}
                          rotationInterval={rotationInterval}
                          onRotationIntervalChange={setRotationInterval}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>
                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <div className="flex items-center gap-2">
                        <b>Advanced Settings</b>
                        <Tooltip
                          title={
                            <span>
                              Learn more about advanced settings in our{" "}
                              <a
                                href={
                                  proxyBaseUrl
                                    ? `${proxyBaseUrl}/#/key%20management/generate_key_fn_key_generate_post`
                                    : `/#/key%20management/generate_key_fn_key_generate_post`
                                }
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-blue-400 hover:text-blue-300"
                              >
                                documentation
                              </a>
                            </span>
                          }
                        >
                          <InfoCircleOutlined className="text-gray-400 hover:text-gray-300 cursor-help" />
                        </Tooltip>
                      </div>
                    </AccordionHeader>
                    <AccordionBody>
                      <SchemaFormFields
                        schemaComponent="GenerateKeyRequest"
                        form={form}
                        excludedFields={[
                          "key_alias",
                          "team_id",
                          "models",
                          "duration",
                          "metadata",
                          "tags",
                          "guardrails",
                          "max_budget",
                          "budget_duration",
                          "tpm_limit",
                          "rpm_limit",
                        ]}
                      />
                    </AccordionBody>
                  </Accordion>
                </AccordionBody>
              </Accordion>
            </div>
          )}

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit" disabled={isFormDisabled} style={{ opacity: isFormDisabled ? 0.5 : 1 }}>
              Create Key
            </Button2>
          </div>
        </Form>
      </Modal>

      {/* Add the Create User Modal */}
      {isCreateUserModalVisible && (
        <Modal
          title="Create New User"
          visible={isCreateUserModalVisible}
          onCancel={() => setIsCreateUserModalVisible(false)}
          footer={null}
          width={800}
        >
          <Createuser
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
            onUserCreated={handleUserCreated}
            isEmbedded={true}
          />
        </Modal>
      )}

      {apiKey && (
        <Modal visible={isModalVisible} onOk={handleOk} onCancel={handleCancel} footer={null}>
          <Grid numItems={1} className="gap-2 w-full">
            <Title>Save your Key</Title>
            <Col numColSpan={1}>
              <p>
                Please save this secret key somewhere safe and accessible. For security reasons,{" "}
                <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key,
                you will need to generate a new one.
              </p>
            </Col>
            <Col numColSpan={1}>
              {apiKey != null ? (
                <div>
                  <Text className="mt-3">API Key:</Text>
                  <div
                    style={{
                      background: "#f8f8f8",
                      padding: "10px",
                      borderRadius: "5px",
                      marginBottom: "10px",
                    }}
                  >
                    <pre style={{ wordWrap: "break-word", whiteSpace: "normal" }}>{apiKey}</pre>
                  </div>

                  <CopyToClipboard text={apiKey} onCopy={handleCopy}>
                    <Button className="mt-3">Copy API Key</Button>
                  </CopyToClipboard>
                  {/* <Button className="mt-3" onClick={sendSlackAlert}>
                    Test Key
                </Button> */}
                </div>
              ) : (
                <Text>Key being created, this might take 30s</Text>
              )}
            </Col>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

export default CreateKey;
