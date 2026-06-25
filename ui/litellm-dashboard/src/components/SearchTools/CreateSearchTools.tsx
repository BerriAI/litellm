import { isAdminRole } from "@/utils/roles";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, TextInput } from "@tremor/react";
import { Form, Input, Modal, Select, Tooltip, Typography } from "antd";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { resolveLogoSrc } from "@/lib/assetPaths";
import NotificationsManager from "../molecules/notifications_manager";
import { createSearchTool, fetchAvailableSearchProviders } from "../networking";
import SearchConnectionTest from "./SearchConnectionTest";
import { AvailableSearchProvider, SearchTool } from "./types";

const { TextArea } = Input;

// Search provider logos folder path (matches existing provider logo pattern)
const searchProviderLogosFolder = "/ui/assets/logos/";

// Helper function to get logo path for a search provider
const getSearchProviderLogo = (providerName: string): string => {
  return `${searchProviderLogosFolder}${providerName}.png`;
};

// Component to display search provider logo and name
interface SearchProviderLabelProps {
  providerName: string;
  displayName: string;
}

const SearchProviderLabel: React.FC<SearchProviderLabelProps> = ({ providerName, displayName }) => (
  <div style={{ display: "flex", alignItems: "center" }}>
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img
      src={resolveLogoSrc(getSearchProviderLogo(providerName))}
      alt=""
      style={{
        width: "20px",
        height: "20px",
        marginRight: "8px",
        objectFit: "contain",
      }}
      onError={(e) => {
        e.currentTarget.style.display = "none";
      }}
    />
    <span>{displayName}</span>
  </div>
);

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
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [isTestModalVisible, setIsTestModalVisible] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  // Fetch available search providers
  const { data: providersResponse, isLoading: isLoadingProviders } = useQuery({
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
      // Prepare the payload
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

      console.log(`Creating search tool with payload:`, payload);

      if (accessToken != null) {
        const response = await createSearchTool(accessToken, payload);

        NotificationsManager.success(t("searchTools.createSearchTools.createSuccess"));
        form.resetFields();
        setFormValues({});
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      NotificationsManager.error(t("searchTools.createSearchTools.createError", { error }));
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
      // Validate required fields for testing
      await form.validateFields(["search_provider", "api_key"]);

      setIsTestingConnection(true);
      // Generate a new test ID (using timestamp for uniqueness)
      setConnectionTestId(`test-${Date.now()}`);
      // Show the modal with the fresh test
      setIsTestModalVisible(true);
    } catch (error) {
      NotificationsManager.error(t("searchTools.createSearchTools.testFillRequired"));
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
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          <span className="text-2xl">🔍</span>
          <h2 className="text-xl font-semibold text-gray-900">{t("searchTools.createSearchTools.modalTitle")}</h2>
        </div>
      }
      open={isModalVisible}
      width={800}
      onCancel={handleCancel}
      footer={null}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-6">
        <Form
          form={form}
          onFinish={handleCreate}
          onValuesChange={(_, allValues) => setFormValues(allValues)}
          layout="vertical"
          className="space-y-6"
        >
          <div className="grid grid-cols-1 gap-6">
            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {t("searchTools.createSearchTools.searchToolNameLabel")}
                  <Tooltip title={t("searchTools.createSearchTools.searchToolNameTooltip")}>
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="search_tool_name"
              rules={[
                { required: true, message: t("searchTools.createSearchTools.searchToolNameRequired") },
                {
                  pattern: /^[a-zA-Z0-9_-]+$/,
                  message: t("searchTools.createSearchTools.searchToolNamePattern"),
                },
              ]}
            >
              <TextInput
                placeholder={t("searchTools.createSearchTools.searchToolNamePlaceholder")}
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {t("searchTools.createSearchTools.searchProviderLabel")}
                  <Tooltip title={t("searchTools.createSearchTools.searchProviderTooltip")}>
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="search_provider"
              rules={[{ required: true, message: t("searchTools.createSearchTools.searchProviderRequired") }]}
            >
              <Select
                placeholder={t("searchTools.createSearchTools.searchProviderPlaceholder")}
                className="rounded-lg"
                size="large"
                loading={isLoadingProviders}
                showSearch
                optionFilterProp="children"
                optionLabelProp="label"
              >
                {availableProviders.map((provider) => (
                  <Select.Option
                    key={provider.provider_name}
                    value={provider.provider_name}
                    label={
                      <SearchProviderLabel
                        providerName={provider.provider_name}
                        displayName={provider.ui_friendly_name}
                      />
                    }
                  >
                    <SearchProviderLabel
                      providerName={provider.provider_name}
                      displayName={provider.ui_friendly_name}
                    />
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {t("searchTools.createSearchTools.apiKeyLabel")}
                  <Tooltip title={t("searchTools.createSearchTools.apiKeyTooltip")}>
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="api_key"
              rules={[{ required: false, message: t("searchTools.createSearchTools.apiKeyRequired") }]}
            >
              <TextInput
                type="password"
                placeholder={t("searchTools.createSearchTools.apiKeyPlaceholder")}
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700">
                  {t("searchTools.createSearchTools.descriptionLabel")}
                </span>
              }
              name="description"
            >
              <TextArea
                rows={3}
                placeholder={t("searchTools.createSearchTools.descriptionPlaceholder")}
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>
          </div>

          <div className="flex justify-between items-center pt-6 border-t border-gray-100">
            <Tooltip title={t("searchTools.createSearchTools.needHelpTooltip")}>
              <Typography.Link href="https://github.com/BerriAI/litellm/issues" target="_blank">
                {t("searchTools.createSearchTools.needHelp")}
              </Typography.Link>
            </Tooltip>
            <div className="space-x-2">
              <Button onClick={handleTestConnection} loading={isTestingConnection}>
                {t("searchTools.createSearchTools.testConnectionBtn")}
              </Button>
              <Button loading={isLoading} type="submit">
                {t("searchTools.createSearchTools.addSearchToolBtn")}
              </Button>
            </div>
          </div>
        </Form>
      </div>

      {/* Test Connection Results Modal */}
      <Modal
        title={t("searchTools.createSearchTools.connectionTestResultsTitle")}
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
            {t("common.close")}
          </Button>,
        ]}
        width={700}
      >
        {/* Only render the SearchConnectionTest when modal is visible and we have a test ID */}
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
