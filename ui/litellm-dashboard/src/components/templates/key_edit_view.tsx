import React, { useState, useEffect } from "react";
import { Form, Input, Select, Button as AntdButton, Tooltip } from "antd";
import { Button as TremorButton, TextInput } from "@tremor/react";
import { KeyResponse } from "../key_team_helpers/key_list";
import { fetchTeamModels } from "../organisms/create_key_button";
import { modelAvailableCall, getPromptsList } from "../networking";
import NumericalInput from "../shared/numerical_input";
import VectorStoreSelector from "../vector_store_management/VectorStoreSelector";
import MCPServerSelector from "../mcp_server_management/MCPServerSelector";
import MCPToolPermissions from "../mcp_server_management/MCPToolPermissions";
import EditLoggingSettings from "../team/EditLoggingSettings";
import { extractLoggingSettings, formatMetadataForDisplay } from "../key_info_utils";
import { fetchMCPAccessGroups } from "../networking";
import { mapInternalToDisplayNames } from "../callback_info_helpers";
import GuardrailSelector from "@/components/guardrails/GuardrailSelector";
import KeyLifecycleSettings from "../common_components/KeyLifecycleSettings";
import RateLimitTypeFormItem from "../common_components/RateLimitTypeFormItem";
import PassThroughRoutesSelector from "../common_components/PassThroughRoutesSelector";

interface KeyEditViewProps {
  keyData: KeyResponse;
  onCancel: () => void;
  onSubmit: (values: any) => Promise<void>;
  teams?: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  premiumUser?: boolean;
}

// Add this helper function
const getAvailableModelsForKey = (keyData: KeyResponse, teams: any[] | null): string[] => {
  // If no teams data is available, return empty array
  console.log("getAvailableModelsForKey:", teams);
  if (!teams || !keyData.team_id) {
    return [];
  }

  // Find the team that matches the key's team_id
  const keyTeam = teams.find((team) => team.team_id === keyData.team_id);

  // If team found and has models, return those models
  if (keyTeam?.models) {
    return keyTeam.models;
  }

  return [];
};

export function KeyEditView({
  keyData,
  onCancel,
  onSubmit,
  teams,
  accessToken,
  userID,
  userRole,
  premiumUser = false,
}: KeyEditViewProps) {
  const [form] = Form.useForm();
  const [userModels, setUserModels] = useState<string[]>([]);
  const [promptsList, setPromptsList] = useState<string[]>([]);
  const team = teams?.find((team) => team.team_id === keyData.team_id);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [mcpAccessGroupsLoaded, setMcpAccessGroupsLoaded] = useState(false);
  const [disabledCallbacks, setDisabledCallbacks] = useState<string[]>(
    Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
      ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
      : [],
  );
  const [autoRotationEnabled, setAutoRotationEnabled] = useState<boolean>(keyData.auto_rotate || false);
  const [rotationInterval, setRotationInterval] = useState<string>(keyData.rotation_interval || "");

  const fetchMcpAccessGroups = async () => {
    if (!accessToken) return;
    if (mcpAccessGroupsLoaded) return;
    try {
      const groups = await fetchMCPAccessGroups(accessToken);
      setMcpAccessGroups(groups);
      setMcpAccessGroupsLoaded(true);
    } catch (error) {
      console.error("Failed to fetch MCP access groups:", error);
    }
  };

  useEffect(() => {
    const fetchModels = async () => {
      if (!userID || !userRole || !accessToken) return;

      try {
        if (keyData.team_id === null) {
          // Fetch user models if no team
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          const available_model_names = model_available["data"].map((element: { id: string }) => element.id);
          setAvailableModels(available_model_names);
        } else if (team?.team_id) {
          // Fetch team models if team exists
          const models = await fetchTeamModels(userID, userRole, accessToken, team.team_id);
          setAvailableModels(Array.from(new Set([...team.models, ...models])));
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    const fetchPrompts = async () => {
      if (!accessToken) return;
      try {
        const response = await getPromptsList(accessToken);
        setPromptsList(response.prompts.map((prompt) => prompt.prompt_id));
      } catch (error) {
        console.error("Failed to fetch prompts:", error);
      }
    };

    fetchPrompts();
    fetchModels();
  }, [userID, userRole, accessToken, team, keyData.team_id]);

  // Sync disabled callbacks with form when component mounts
  useEffect(() => {
    form.setFieldValue("disabled_callbacks", disabledCallbacks);
  }, [form, disabledCallbacks]);

  // Convert API budget duration to form format
  const getBudgetDuration = (duration: string | null) => {
    if (!duration) return null;
    const durationMap: Record<string, string> = {
      "24h": "daily",
      "7d": "weekly",
      "30d": "monthly",
    };
    return durationMap[duration] || null;
  };

  // Set initial form values
  const initialValues = {
    ...keyData,
    token: keyData.token || keyData.token_id,
    budget_duration: getBudgetDuration(keyData.budget_duration),
    metadata: formatMetadataForDisplay(keyData.metadata),
    guardrails: keyData.metadata?.guardrails,
    prompts: keyData.metadata?.prompts,
    vector_stores: keyData.object_permission?.vector_stores || [],
    mcp_servers_and_groups: {
      servers: keyData.object_permission?.mcp_servers || [],
      accessGroups: keyData.object_permission?.mcp_access_groups || [],
    },
    mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
    logging_settings: extractLoggingSettings(keyData.metadata),
    disabled_callbacks: Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
      ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
      : [],
    auto_rotate: keyData.auto_rotate || false,
    ...(keyData.rotation_interval && { rotation_interval: keyData.rotation_interval }),
  };

  useEffect(() => {
    form.setFieldsValue({
      ...keyData,
      token: keyData.token || keyData.token_id,
      budget_duration: getBudgetDuration(keyData.budget_duration),
      metadata: formatMetadataForDisplay(keyData.metadata),
      guardrails: keyData.metadata?.guardrails,
      prompts: keyData.metadata?.prompts,
      vector_stores: keyData.object_permission?.vector_stores || [],
      mcp_servers_and_groups: {
        servers: keyData.object_permission?.mcp_servers || [],
        accessGroups: keyData.object_permission?.mcp_access_groups || [],
      },
      mcp_tool_permissions: keyData.object_permission?.mcp_tool_permissions || {},
      logging_settings: extractLoggingSettings(keyData.metadata),
      disabled_callbacks: Array.isArray(keyData.metadata?.litellm_disabled_callbacks)
        ? mapInternalToDisplayNames(keyData.metadata.litellm_disabled_callbacks)
        : [],
      auto_rotate: keyData.auto_rotate || false,
      ...(keyData.rotation_interval && { rotation_interval: keyData.rotation_interval }),
    });
  }, [keyData, form]);

  // Sync auto-rotation state with form values
  useEffect(() => {
    form.setFieldValue("auto_rotate", autoRotationEnabled);
  }, [autoRotationEnabled, form]);

  useEffect(() => {
    if (rotationInterval) {
      form.setFieldValue("rotation_interval", rotationInterval);
    }
  }, [rotationInterval, form]);

  console.log("premiumUser:", premiumUser);

  return (
    <Form form={form} onFinish={onSubmit} initialValues={initialValues} layout="vertical">
      <Form.Item label="Key Alias" name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item label="Models" name="models">
        <Select mode="multiple" placeholder="Select models" style={{ width: "100%" }}>
          {/* Only show All Team Models if team has models */}
          {availableModels.length > 0 && <Select.Option value="all-team-models">All Team Models</Select.Option>}
          {/* Show available team models */}
          {availableModels.map((model) => (
            <Select.Option key={model} value={model}>
              {model}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item label="Max Budget (USD)" name="max_budget">
        <NumericalInput step={0.01} style={{ width: "100%" }} placeholder="Enter a numerical value" />
      </Form.Item>

      <Form.Item label="Reset Budget" name="budget_duration">
        <Select placeholder="n/a">
          <Select.Option value="daily">Daily</Select.Option>
          <Select.Option value="weekly">Weekly</Select.Option>
          <Select.Option value="monthly">Monthly</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item label="TPM Limit" name="tpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="tpm" name="tpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item label="RPM Limit" name="rpm_limit">
        <NumericalInput min={0} />
      </Form.Item>

      <RateLimitTypeFormItem type="rpm" name="rpm_limit_type" showDetailedDescriptions={false} />

      <Form.Item label="Max Parallel Requests" name="max_parallel_requests">
        <NumericalInput min={0} />
      </Form.Item>

      <Form.Item label="Model TPM Limit" name="model_tpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item label="Model RPM Limit" name="model_rpm_limit">
        <Input.TextArea rows={4} placeholder='{"gpt-4": 100, "claude-v1": 200}' />
      </Form.Item>

      <Form.Item label="Guardrails" name="guardrails">
        {accessToken && (
          <GuardrailSelector
            onChange={(v) => {
              form.setFieldValue("guardrails", v);
            }}
            accessToken={accessToken}
            disabled={!premiumUser}
          />
        )}
      </Form.Item>

      <Form.Item label="Prompts" name="prompts">
        <Tooltip title={!premiumUser ? "Setting prompts by key is a premium feature" : ""} placement="top">
          <Select
            mode="tags"
            style={{ width: "100%" }}
            disabled={!premiumUser}
            placeholder={
              !premiumUser
                ? "Premium feature - Upgrade to set prompts by key"
                : Array.isArray(keyData.metadata?.prompts) && keyData.metadata.prompts.length > 0
                  ? `Current: ${keyData.metadata.prompts.join(", ")}`
                  : "Select or enter prompts"
            }
            options={promptsList.map((name) => ({ value: name, label: name }))}
          />
        </Tooltip>
      </Form.Item>

      <Form.Item label="Allowed Pass Through Routes" name="allowed_passthrough_routes">
        <Tooltip title={!premiumUser ? "Setting allowed pass through routes by key is a premium feature" : ""} placement="top">
          <PassThroughRoutesSelector
            onChange={(values: string[]) => form.setFieldValue("allowed_passthrough_routes", values)}
            value={form.getFieldValue("allowed_passthrough_routes")}
            accessToken={accessToken || ""}
            placeholder={
              !premiumUser
                ? "Premium feature - Upgrade to set allowed pass through routes by key"
                : Array.isArray(keyData.metadata?.allowed_passthrough_routes) && keyData.metadata.allowed_passthrough_routes.length > 0
                  ? `Current: ${keyData.metadata.allowed_passthrough_routes.join(", ")}`
                  : "Select or enter allowed pass through routes"
            }
            disabled={!premiumUser}
          />
        </Tooltip>
      </Form.Item>

      <Form.Item label="Vector Stores" name="vector_stores">
        <VectorStoreSelector
          onChange={(values: string[]) => form.setFieldValue("vector_stores", values)}
          value={form.getFieldValue("vector_stores")}
          accessToken={accessToken || ""}
          placeholder="Select vector stores"
        />
      </Form.Item>

      <Form.Item label="MCP Servers / Access Groups" name="mcp_servers_and_groups">
        <MCPServerSelector
          onChange={(val) => form.setFieldValue("mcp_servers_and_groups", val)}
          value={form.getFieldValue("mcp_servers_and_groups")}
          accessToken={accessToken || ""}
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
          prevValues.mcp_servers_and_groups !== currentValues.mcp_servers_and_groups ||
          prevValues.mcp_tool_permissions !== currentValues.mcp_tool_permissions
        }
      >
        {() => (
          <div className="mb-6">
            <MCPToolPermissions
              accessToken={accessToken || ""}
              selectedServers={form.getFieldValue("mcp_servers_and_groups")?.servers || []}
              toolPermissions={form.getFieldValue("mcp_tool_permissions") || {}}
              onChange={(toolPerms) => form.setFieldsValue({ mcp_tool_permissions: toolPerms })}
            />
          </div>
        )}
      </Form.Item>

      <Form.Item label="Team ID" name="team_id">
        <Select placeholder="Select team" style={{ width: "100%" }}>
          {/* Only show All Team Models if team has models */}
          {teams?.map((team) => (
            <Select.Option key={team.team_id} value={team.team_id}>
              {`${team.team_alias} (${team.team_id})`}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>
      <Form.Item label="Logging Settings" name="logging_settings">
        <EditLoggingSettings
          value={form.getFieldValue("logging_settings")}
          onChange={(values) => form.setFieldValue("logging_settings", values)}
          disabledCallbacks={disabledCallbacks}
          onDisabledCallbacksChange={(internalValues) => {
            // Convert internal values back to display names for UI state
            const displayNames = mapInternalToDisplayNames(internalValues);
            setDisabledCallbacks(displayNames);
            // Store internal values in form for submission
            form.setFieldValue("disabled_callbacks", internalValues);
          }}
        />
      </Form.Item>

      <Form.Item label="Metadata" name="metadata">
        <Input.TextArea rows={10} />
      </Form.Item>

      {/* Auto-Rotation Settings */}
      <div className="mb-4">
        <KeyLifecycleSettings
          form={form}
          autoRotationEnabled={autoRotationEnabled}
          onAutoRotationChange={setAutoRotationEnabled}
          rotationInterval={rotationInterval}
          onRotationIntervalChange={setRotationInterval}
        />
      </div>

      {/* Hidden form field for token */}
      <Form.Item name="token" hidden>
        <Input />
      </Form.Item>

      {/* Hidden form field for disabled callbacks */}
      <Form.Item name="disabled_callbacks" hidden>
        <Input />
      </Form.Item>

      {/* Hidden form fields for auto-rotation */}
      <Form.Item name="auto_rotate" hidden>
        <Input />
      </Form.Item>
      <Form.Item name="rotation_interval" hidden>
        <Input />
      </Form.Item>

      <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
        <div className="flex justify-end items-center gap-2">
          <AntdButton onClick={onCancel}>Cancel</AntdButton>
          <TremorButton type="submit">Save Changes</TremorButton>
        </div>
      </div>
    </Form>
  );
}
