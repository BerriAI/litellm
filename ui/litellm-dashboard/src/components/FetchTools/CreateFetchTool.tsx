import { LoadingOutlined } from "@ant-design/icons";
import { Form, Input, Modal, Select, Spin } from "antd";
import React, { useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { createFetchTool } from "../networking";
import { AvailableFetchProvider, FetchTool } from "./types";

interface CreateFetchToolProps {
  accessToken: string | null;
  availableProviders: AvailableFetchProvider[];
  onSuccess: (newFetchTool: FetchTool) => void;
  onCancel: () => void;
}

const CreateFetchTool: React.FC<CreateFetchToolProps> = ({ accessToken, availableProviders, onSuccess, onCancel }) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (!accessToken) return;

    try {
      const values = await form.validateFields();
      setIsLoading(true);

      const fetchToolData = {
        fetch_tool_name: values.fetch_tool_name,
        litellm_params: {
          fetch_provider: values.fetch_provider,
          api_key: values.api_key,
          api_base: values.api_base,
          timeout: values.timeout ? parseFloat(values.timeout) : undefined,
          max_retries: values.max_retries ? parseInt(values.max_retries) : undefined,
        },
        fetch_tool_info: values.description
          ? {
              description: values.description,
            }
          : undefined,
      };

      const result = await createFetchTool(accessToken, fetchToolData);
      NotificationsManager.success("Fetch tool created successfully");
      form.resetFields();
      onSuccess(result);
    } catch (error) {
      console.error("Failed to create fetch tool:", error);
      NotificationsManager.error("Failed to create fetch tool");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Spin spinning={isLoading} indicator={<LoadingOutlined spin />} size="large">
      <Form form={form} layout="vertical">
        <Form.Item
          name="fetch_tool_name"
          label="Fetch Tool Name"
          rules={[{ required: true, message: "Please enter a fetch tool name" }]}
        >
          <Input placeholder="e.g., my-firecrawl-fetch" />
        </Form.Item>

        <Form.Item
          name="fetch_provider"
          label="Fetch Provider"
          rules={[{ required: true, message: "Please select a fetch provider" }]}
        >
          <Select placeholder="Select a fetch provider">
            {availableProviders.map((provider) => (
              <Select.Option key={provider.provider_name} value={provider.provider_name}>
                {provider.ui_friendly_name}
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item name="api_key" label="API Key" extra="API key for the fetch provider">
          <Input.Password placeholder="Enter API key" />
        </Form.Item>

        <Form.Item name="api_base" label="API Base URL" extra="Optional: Custom API base URL for self-hosted instances">
          <Input placeholder="https://api.firecrawl.dev" />
        </Form.Item>

        <Form.Item name="timeout" label="Timeout (seconds)">
          <Input type="number" placeholder="30" />
        </Form.Item>

        <Form.Item name="max_retries" label="Max Retries">
          <Input type="number" placeholder="3" />
        </Form.Item>

        <Form.Item name="description" label="Description">
          <Input.TextArea rows={3} placeholder="Description of this fetch tool" />
        </Form.Item>

        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onCancel} className="px-4 py-2 border rounded hover:bg-gray-50" disabled={isLoading}>
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
            disabled={isLoading}
          >
            Create Fetch Tool
          </button>
        </div>
      </Form>
    </Spin>
  );
};

export default CreateFetchTool;
