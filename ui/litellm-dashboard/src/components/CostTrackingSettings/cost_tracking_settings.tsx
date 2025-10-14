import React, { useState, useEffect } from "react";
import { Card, Title, Text, Subtitle, Grid, Col, Button } from "@tremor/react";
import { Modal, Form } from "antd";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Providers, provider_map } from "../provider_info_helpers";
import { CostTrackingSettingsProps, DiscountConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";
import { ExclamationCircleOutlined } from "@ant-design/icons";

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

  useEffect(() => {
    if (accessToken) {
      fetchDiscountConfig();
    }
  }, [accessToken]);

  const fetchDiscountConfig = async () => {
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
  };

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
          <Title>Cost Tracking Settings</Title>
          <Text className="text-gray-500">
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

      {/* Main Content Card */}
      <div className="bg-white rounded-lg shadow w-full max-w-full">
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
          <div className="py-16 text-center">
            <Text className="text-gray-500">
              No provider discounts configured. Click "Add Provider Discount" to get started.
            </Text>
          </div>
        )}

        <div className="border-t px-6 py-4 bg-gray-50">
          <Title className="mb-3">How It Works</Title>
          <div className="space-y-2">
            <div>
              <Text className="font-medium text-gray-900 text-sm">Cost Calculation</Text>
              <Text className="text-xs text-gray-600">
                Discounts are applied to provider costs: <code className="bg-gray-200 px-1.5 py-0.5 rounded text-xs">final_cost = base_cost × (1 - discount%/100)</code>
              </Text>
            </div>
            <div>
              <Text className="font-medium text-gray-900 text-sm">Example</Text>
              <Text className="text-xs text-gray-600">
                A 5% discount on a $10.00 request results in: $10.00 × (1 - 0.05) = $9.50
              </Text>
            </div>
            <div>
              <Text className="font-medium text-gray-900 text-sm">Valid Range</Text>
              <Text className="text-xs text-gray-600">
                Discount percentages must be between 0% and 100%
              </Text>
            </div>
          </div>
        </div>
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
