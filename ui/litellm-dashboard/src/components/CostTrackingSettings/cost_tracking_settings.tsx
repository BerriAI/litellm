import React, { useState, useEffect, useCallback } from "react";
import { Title, Text, Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Modal, Form } from "antd";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Providers } from "../provider_info_helpers";
import { CostTrackingSettingsProps, DiscountConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";
import { ExclamationCircleOutlined } from "@ant-design/icons";
import { DocsMenu } from "../HelpLink";
import HowItWorks from "./how_it_works";

const DOCS_LINKS = [
  { label: "Custom pricing for models", href: "https://docs.litellm.ai/docs/proxy/custom_pricing" },
  { label: "Spend tracking", href: "https://docs.litellm.ai/docs/proxy/cost_tracking" },
];

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ 
  userID, 
  userRole, 
  accessToken 
}) => {
  const [discountConfig, setDiscountConfig] = useState<DiscountConfig>({});
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>(undefined);
  const [newDiscount, setNewDiscount] = useState<string>("");
  const [isFetching, setIsFetching] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [modal, contextHolder] = Modal.useModal();

  const fetchDiscountConfig = useCallback(async () => {
    setIsFetching(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_discount_config` 
        : "/config/cost_discount_config";
      
      const response = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const data = await response.json();
        setDiscountConfig(data.values || {});
      } else {
        console.error("Failed to fetch discount config");
      }
    } catch (error) {
      console.error("Error fetching discount config:", error);
      NotificationsManager.fromBackend("Failed to fetch discount configuration");
    } finally {
      setIsFetching(false);
    }
  }, [accessToken]);

  useEffect(() => {
    if (accessToken) {
      fetchDiscountConfig();
    }
  }, [accessToken, fetchDiscountConfig]);

  const saveDiscountConfig = async (config: DiscountConfig) => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl 
        ? `${proxyBaseUrl}/config/cost_discount_config` 
        : "/config/cost_discount_config";
      
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        NotificationsManager.success("Discount configuration updated successfully");
        await fetchDiscountConfig();
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.detail?.error || errorData.detail || "Failed to update settings";
        NotificationsManager.fromBackend(errorMessage);
      }
    } catch (error) {
      console.error("Error updating discount config:", error);
      NotificationsManager.fromBackend("Failed to update discount configuration");
    }
  };

  const handleAddProvider = async () => {
    if (!selectedProvider || !newDiscount) {
      NotificationsManager.fromBackend("Please select a provider and enter discount percentage");
      return;
    }
    
    const percentageValue = parseFloat(newDiscount);
    if (isNaN(percentageValue) || percentageValue < 0 || percentageValue > 100) {
      NotificationsManager.fromBackend("Discount must be between 0% and 100%");
      return;
    }

    const providerValue = getProviderBackendValue(selectedProvider);
    
    if (!providerValue) {
      NotificationsManager.fromBackend("Invalid provider selected");
      return;
    }

    if (discountConfig[providerValue]) {
      NotificationsManager.fromBackend(
        `Discount for ${Providers[selectedProvider as keyof typeof Providers]} already exists. Edit it in the table above.`
      );
      return;
    }

    // Convert percentage to decimal for storage
    const discountValue = percentageValue / 100;
    const updatedConfig = {
      ...discountConfig,
      [providerValue]: discountValue,
    };
    
    setDiscountConfig(updatedConfig);
    await saveDiscountConfig(updatedConfig);
    setSelectedProvider(undefined);
    setNewDiscount("");
    setIsModalVisible(false);
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedProvider(undefined);
    setNewDiscount("");
  };

  const handleFormSubmit = (values: any) => {
    handleAddProvider();
  };

  const handleRemoveProvider = async (provider: string, providerDisplayName: string) => {
    modal.confirm({
      title: 'Remove Provider Discount',
      icon: <ExclamationCircleOutlined />,
      content: `Are you sure you want to remove the discount for ${providerDisplayName}?`,
      okText: 'Remove',
      okType: 'danger',
      cancelText: 'Cancel',
      onOk: async () => {
        const updatedConfig = { ...discountConfig };
        delete updatedConfig[provider];
        setDiscountConfig(updatedConfig);
        await saveDiscountConfig(updatedConfig);
      },
    });
  };

  const handleDiscountChange = async (provider: string, value: string) => {
    const discountValue = parseFloat(value);
    if (!isNaN(discountValue) && discountValue >= 0 && discountValue <= 1) {
      const updatedConfig = {
        ...discountConfig,
        [provider]: discountValue,
      };
      setDiscountConfig(updatedConfig);
      await saveDiscountConfig(updatedConfig);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full p-8">
      {contextHolder}
      
      {/* Header Section - Outside the card */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <Title>Cost Tracking Settings</Title>
            <DocsMenu items={DOCS_LINKS} />
          </div>
          <Text className="text-gray-500 mt-1">
            Configure cost discounts for different LLM providers. Changes are saved automatically.
          </Text>
        </div>
        <Button
          onClick={() => setIsModalVisible(true)}
          className="mt-4 md:mt-0"
        >
          + Add Provider Discount
        </Button>
      </div>

      {/* Main Content Card with Tabs */}
      <div className="bg-white rounded-lg shadow w-full max-w-full">
        <TabGroup>
          <TabList className="px-6 pt-4">
            <Tab>Provider Discounts</Tab>
            <Tab>Test It</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              {isFetching ? (
                <div className="py-12 text-center">
                  <Text className="text-gray-500">Loading configuration...</Text>
                </div>
              ) : Object.keys(discountConfig).length > 0 ? (
                <div className="p-6">
                  <ProviderDiscountTable
                    discountConfig={discountConfig}
                    onDiscountChange={handleDiscountChange}
                    onRemoveProvider={handleRemoveProvider}
                  />
                </div>
              ) : (
                <div className="py-16 px-6 text-center">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400 mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <Text className="text-gray-700 font-medium mb-2">
                    No provider discounts configured
                  </Text>
                  <Text className="text-gray-500 text-sm">
                    Click &quot;Add Provider Discount&quot; to get started
                  </Text>
                </div>
              )}
            </TabPanel>
            <TabPanel>
              <div className="px-6 pb-4">
                <HowItWorks />
              </div>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </div>

      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900">Add Provider Discount</h2>
          </div>
        }
        open={isModalVisible}
        width={1000}
        onCancel={handleModalCancel}
        footer={null}
        className="top-8"
        styles={{
          body: { padding: "24px" },
          header: { padding: "24px 24px 0 24px", border: "none" },
        }}
      >
        <div className="mt-6">
          <Text className="text-sm text-gray-600 mb-6">
            Select a provider and set its discount percentage. Enter a value between 0% and 100% (e.g., 5 for a 5% discount).
          </Text>
          <Form
            form={form}
            onFinish={handleFormSubmit}
            layout="vertical"
            className="space-y-6"
          >
            <AddProviderForm
              discountConfig={discountConfig}
              selectedProvider={selectedProvider}
              newDiscount={newDiscount}
              onProviderChange={setSelectedProvider}
              onDiscountChange={setNewDiscount}
              onAddProvider={handleAddProvider}
            />
          </Form>
        </div>
      </Modal>
    </div>
  );
};

export default CostTrackingSettings;
