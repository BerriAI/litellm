import { isAdminRole } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Form, Input, Modal, Select, Tooltip, Typography } from "antd";
import React, { useState } from "react";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import NotificationsManager from "../molecules/notifications_manager";
import { createSearchTool, fetchAvailableSearchProviders } from "../networking";
import SearchConnectionTest from "./SearchConnectionTest";
import { AvailableSearchProvider, SearchTool } from "./types";

interface CreateSearchToolProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newSearchTool: SearchTool) => void;
  isModalVisible: boolean;
  setModalVisible: (visible: boolean) => void;
}

const CreateSearchTool: React.FC<CreateSearchToolProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
}) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [isTestModalVisible, setIsTestModalVisible] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  // Fetch available search providers
  const {
    data: providersResponse,
    isLoading: isLoadingProviders,
  } = useQuery({
    queryKey: ["searchProviders"],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return fetchAvailableSearchProviders(accessToken);
    },
    enabled: !!accessToken && isModalVisible,
  }) as { data: { providers: AvailableSearchProvider[] }; isLoading: boolean };

  const availableProviders = providersResponse?.providers || [];

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      const payload = {
        search_tool_name: formValues.search_tool_name,
        litellm_params: {
          search_provider: formValues.search_provider,
          api_key: formValues.api_key,
          api_base: formValues.api_base,
          timeout: formValues.timeout ? parseFloat(formValues.timeout) : undefined,
          max_retries: formValues.max_retries ? parseInt(formValues.max_retries) : undefined,
        },
        search_tool_info: formValues.description
          ? {
            description: formValues.description,
          }
          : undefined,
      };

      if (accessToken != null) {
        const response = await createSearchTool(accessToken, payload);

        NotificationsManager.success("Search tool created successfully");
        form.resetFields();
        setFormValues({});
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      NotificationsManager.error("Error creating search tool: " + error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    setFormValues({});
    setModalVisible(false);
  };

  const handleTestConnection = async () => {
    try {
      await form.validateFields(["search_provider", "api_key"]);

      setIsTestingConnection(true);
      setConnectionTestId(`test-${Date.now()}`);
      setIsTestModalVisible(true);
    } catch (error) {
      NotificationsManager.error("Please fill in Search Provider and API Key before testing");
    }
  };

  // Clear formValues when modal closes to reset
  React.useEffect(() => {
    if (!isModalVisible) {
      setFormValues({});
    }
  }, [isModalVisible]);

  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <Modal
      title="Add New Search Tool"
      open={isModalVisible}
      width={600}
      onCancel={handleCancel}
      footer={
        <div className="flex justify-between items-center">
          <Typography.Link href="https://github.com/BerriAI/litellm/issues" target="_blank">
            Need Help?
          </Typography.Link>
          <div className="space-x-2">
            <Button onClick={handleTestConnection} loading={isTestingConnection}>
              Test Connection
            </Button>
            <Button type="primary" onClick={() => form.submit()} loading={isLoading}>
              Add Search Tool
            </Button>
          </div>
        </div>
      }
    >
      <Form
        form={form}
        onFinish={handleCreate}
        onValuesChange={(_, allValues) => setFormValues(allValues)}
        layout="vertical"
        className="mt-4"
      >
        <Form.Item
          label={
            <span>
              Search Tool Name{" "}
              <Tooltip title="A unique name to identify this search tool configuration (e.g., 'perplexity-search', 'tavily-news-search').">
                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
              </Tooltip>
            </span>
          }
          name="search_tool_name"
          rules={[
            { required: true, message: "Please enter a search tool name" },
            {
              pattern: /^[a-zA-Z0-9_-]+$/,
              message: "Name can only contain letters, numbers, hyphens, and underscores",
            },
          ]}
        >
          <Input placeholder="e.g., perplexity-search, my-tavily-tool" />
        </Form.Item>

        <Form.Item
          label={
            <span>
              Search Provider{" "}
              <Tooltip title="Select the search provider you want to use. Each provider has different capabilities and pricing.">
                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
              </Tooltip>
            </span>
          }
          name="search_provider"
          rules={[{ required: true, message: "Please select a search provider" }]}
        >
          <Select
            placeholder="Select a search provider"
            showSearch
            optionFilterProp="children"
            optionLabelProp="label"
            notFoundContent={isLoadingProviders ? "Loading providers..." : "No providers found"}
          >
            {availableProviders.map((provider) => (
              <Select.Option
                key={provider.provider_name}
                value={provider.provider_name}
                label={provider.ui_friendly_name}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <ProviderLogo provider={provider.provider_name} className="w-5 h-5" />
                  <span>{provider.ui_friendly_name}</span>
                </div>
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          label={
            <span>
              API Key{" "}
              <Tooltip title="The API key for authenticating with the search provider. This will be securely stored.">
                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
              </Tooltip>
            </span>
          }
          name="api_key"
        >
          <Input.Password placeholder="Enter your API key" />
        </Form.Item>

        <Form.Item
          label="Description (Optional)"
          name="description"
        >
          <Input.TextArea
            rows={3}
            placeholder="Brief description of this search tool's purpose"
          />
        </Form.Item>
      </Form>

      {/* Test Connection Results Modal */}
      <Modal
        title="Connection Test Results"
        open={isTestModalVisible}
        onCancel={() => {
          setIsTestModalVisible(false);
          setIsTestingConnection(false);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setIsTestModalVisible(false);
              setIsTestingConnection(false);
            }}
          >
            Close
          </Button>,
        ]}
        width={700}
      >
        {isTestModalVisible && accessToken && (
          <SearchConnectionTest
            key={connectionTestId}
            litellmParams={{
              search_provider: formValues.search_provider,
              api_key: formValues.api_key,
              api_base: formValues.api_base,
            }}
            accessToken={accessToken}
            onTestComplete={() => setIsTestingConnection(false)}
          />
        )}
      </Modal>
    </Modal>
  );
};

export default CreateSearchTool;
