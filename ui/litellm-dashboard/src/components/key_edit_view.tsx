import React, { useState, useEffect } from "react";
import { Form, Input, InputNumber, Select } from "antd";
import { Button, TextInput } from "@tremor/react";
import { KeyResponse } from "./key_team_helpers/key_list";
import { getTeamModels } from "../components/create_key_button";
import { modelAvailableCall } from "./networking";

interface KeyEditViewProps {
  keyData: KeyResponse;
  onCancel: () => void;
  onSubmit: (values: any) => Promise<void>;
  teams?: any[] | null;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
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
    userRole }: KeyEditViewProps) {
  const [form] = Form.useForm();
  const [userModels, setUserModels] = useState<string[]>([]);
  const team = teams?.find(team => team.team_id === keyData.team_id);
  const availableModels = getTeamModels(team, userModels);
  

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (accessToken && userID && userRole) {
          const model_available = await modelAvailableCall(
            accessToken,
            userID,
            userRole
          );
          let available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
          console.log("available_model_names:", available_model_names);
          setUserModels(available_model_names);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };

    fetchUserModels();
  }, []);

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
    guardrails: keyData.metadata?.guardrails || []
  };

  return (
    <Form
      form={form}
      onFinish={onSubmit}
      initialValues={initialValues}
      layout="vertical"
    >
      <Form.Item label="Key Alias" name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item label="Models" name="models">
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

      <Form.Item label="Max Budget (USD)" name="max_budget">
        <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="Reset Budget" name="budget_duration">
        <Select placeholder="n/a">
          <Select.Option value="daily">Daily</Select.Option>
          <Select.Option value="weekly">Weekly</Select.Option>
          <Select.Option value="monthly">Monthly</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item label="TPM Limit" name="tpm_limit">
        <InputNumber style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="RPM Limit" name="rpm_limit">
        <InputNumber style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="Guardrails" name="guardrails">
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder="Select or enter guardrails"
        />
      </Form.Item>

      <Form.Item label="Metadata" name="metadata">
        <Input.TextArea rows={10} />
      </Form.Item>

      {/* Hidden form field for token */}
      <Form.Item name="token" hidden>
        <Input />
      </Form.Item>

      <div className="flex justify-end gap-2 mt-6">
        <Button variant="light" onClick={onCancel}>
          Cancel
        </Button>
        <Button>
          Save Changes
        </Button>
      </div>
    </Form>
  );
} 