import React, { useState, useEffect } from "react";
import { Form, Input, Select, Button as AntdButton, Tooltip } from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { Button as TremorButton, TextInput } from "@tremor/react";
import { KeyResponse } from "./key_team_helpers/key_list";
import { fetchTeamModels } from "../components/create_key_button";
import { modelAvailableCall } from "./networking";
import NumericalInput from "./shared/numerical_input";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import MCPServerSelector from "./mcp_server_management/MCPServerSelector";

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
  const keyTeam = teams.find(team => team.team_id === keyData.team_id);
  
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
    premiumUser = false
}: KeyEditViewProps) {
  const [form] = Form.useForm();
  const [userModels, setUserModels] = useState<string[]>([]);
  const team = teams?.find(team => team.team_id === keyData.team_id);
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  useEffect(() => {
    const fetchModels = async () => {
      if (!userID || !userRole || !accessToken) return;

      try {
        if (keyData.team_id === null) {
          // Fetch user models if no team
          const model_available = await modelAvailableCall(
            accessToken,
            userID, 
            userRole
          );
          const available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
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

    fetchModels();
  }, [userID, userRole, accessToken, team, keyData.team_id]);

  // Convert API budget duration to form format
  const getBudgetDuration = (duration: string | null) => {
    if (!duration) return null;
    const durationMap: Record<string, string> = {
      "24h": "daily",
      "7d": "weekly",
      "30d": "monthly"
    };
    return durationMap[duration] || null;
  };

  // Set initial form values
  const initialValues = {
    ...keyData,
    budget_duration: getBudgetDuration(keyData.budget_duration),
    metadata: keyData.metadata ? JSON.stringify(keyData.metadata, null, 2) : "",
    guardrails: keyData.metadata?.guardrails || [],
    vector_stores: keyData.object_permission?.vector_stores || [],
    mcp_servers: keyData.object_permission?.mcp_servers || []
  };

  return (
    <Form
      form={form}
      onFinish={onSubmit}
      initialValues={initialValues}
      layout="vertical"
    >
      <Form.Item 
        label={
          <span>
            Key Alias{' '}
            <Tooltip title="Descriptive name for the key">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Models{' '}
            <Tooltip title="Select which models this key can access. Choose 'All Team Models' to grant access to all models available to the team">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="models">
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
        >
          {/* Only show All Team Models if team has models */}
          {availableModels.length > 0 && (
            <Select.Option value="all-team-models">All Team Models</Select.Option>
          )}
          {/* Show available team models */}
          {availableModels.map(model => (
            <Select.Option key={model} value={model}>
              {model}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Expire Key{' '}
            <Tooltip title="Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days)">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="duration">
        <TextInput style={{ width: "100%" }} placeholder="e.g., 30d"/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Max Budget (USD){' '}
            <Tooltip title="Maximum amount in USD this key can spend. When reached, the key will be blocked from making further requests">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="max_budget">
        <NumericalInput step={0.01} style={{ width: "100%" }} placeholder="Enter a numerical value"/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Reset Budget{' '}
            <Tooltip title="How often the budget should reset. For example, setting 'daily' will reset the budget every 24 hours">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        } 
        name="budget_duration">
        <Select placeholder="n/a">
          <Select.Option value="daily">Daily</Select.Option>
          <Select.Option value="weekly">Weekly</Select.Option>
          <Select.Option value="monthly">Monthly</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Tokens per minute Limit (TPM){' '}
            <Tooltip title="Maximum number of tokens this key can process per minute. Helps control usage and costs">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        } 
        name="tpm_limit">
        <NumericalInput min={0}/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Requests per minute Limit (RPM){' '}
            <Tooltip title="Maximum number of API requests this key can make per minute. Helps prevent abuse and manage load">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="rpm_limit">
        <NumericalInput min={0}/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Max Parallel Requests{' '}
            <Tooltip title="Limit the max concurrent calls made to a deployment.">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        } name="max_parallel_requests">  
        <NumericalInput min={0}/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Model TPM Limit{' '}
            <Tooltip title="Set per-model TPM limits.">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="model_tpm_limit">
        <Input.TextArea rows={4}  placeholder='{"gpt-4": 100, "claude-v1": 200}'/>
      </Form.Item>

      <Form.Item
        label={
          <span>
            Model RPM Limit{' '}
            <Tooltip title="Set per-model RPM limits.">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="model_rpm_limit">
        <Input.TextArea rows={4}  placeholder='{"gpt-4": 100, "claude-v1": 200}'/>
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Guardrails{' '}
            <Tooltip title="Apply safety guardrails to this key to filter content or enforce policies">
              <a 
                href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
                target="_blank" 
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
              >
                <InfoCircleOutlined style={{ marginLeft: '4px' }} />
              </a>
            </Tooltip>
          </span>
        }
        name="guardrails">
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder="Select or enter guardrails"
        />
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Vector Stores{' '}
            <Tooltip title="Select which vector stores this key can access. If none selected, the key will have access to all available vector stores">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="vector_stores">
        <VectorStoreSelector
          onChange={(values) => form.setFieldValue('vector_stores', values)}
          value={form.getFieldValue('vector_stores')}
          accessToken={accessToken || ""}
          placeholder="Select vector stores"
        />
      </Form.Item>

      <Form.Item 
        label={
          <span>
            MCP Servers{' '}
            <Tooltip title="Select which MCP servers this key can access. If none selected, the key will have access to all available MCP servers">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="mcp_servers">
        <MCPServerSelector
          onChange={(values) => form.setFieldValue('mcp_servers', values)}
          value={form.getFieldValue('mcp_servers')}
          accessToken={accessToken || ""}
          placeholder="Select MCP servers"
        />
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Metadata{' '}
            <Tooltip title="JSON object with additional information about this key. Used for tracking or custom logic">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="metadata">
        <Input.TextArea rows={10} />
      </Form.Item>

      <Form.Item 
        label={
          <span>
            Team{' '}
            <Tooltip title="The team this key belongs to, which determines available models and budget limits">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="team_id">
        <Select
          placeholder="Select team"
          style={{ width: "100%" }}
        >
          {/* Only show All Team Models if team has models */}
          {teams?.map(team => (
            <Select.Option key={team.team_id} value={team.team_id}>
              {`${team.team_alias} (${team.team_id})`}
            </Select.Option>
          ))}
        </Select>
      </Form.Item>

      {/* Hidden form field for token */}
      <Form.Item name="token" hidden>
        <Input />
      </Form.Item>

      <div className="sticky z-10 bg-white p-4 border-t border-gray-200 bottom-[-1.5rem] inset-x-[-1.5rem]">
        <div className="flex justify-end items-center gap-2">
          <AntdButton onClick={onCancel}>
            Cancel
          </AntdButton>
          <TremorButton type="submit">
            Save Changes
          </TremorButton>
        </div>
      </div>
    </Form>
  );
} 