import React, { useState, useEffect } from "react";
import { Card, Title, Text, Subtitle, Grid, Col, Button } from "@tremor/react";
import { Modal } from "antd";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Providers, provider_map } from "../provider_info_helpers";
import { CostTrackingSettingsProps, DiscountConfig } from "./types";
import { getProviderBackendValue } from "./provider_display_helpers";
import ProviderDiscountTable from "./provider_discount_table";
import AddProviderForm from "./add_provider_form";

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ 
  userID, 
  userRole, 
  accessToken 
}) => {
  const [discountConfig, setDiscountConfig] = useState<DiscountConfig>({});
  const [selectedProvider, setSelectedProvider] = useState<string | undefined>(undefined);
  const [newDiscount, setNewDiscount] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [isModalVisible, setIsModalVisible] = useState(false);

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

  const handleSave = async () => {
    setLoading(true);
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
        body: JSON.stringify(discountConfig),
      });

      if (response.ok) {
        NotificationsManager.success("Cost discount configuration updated successfully");
        await fetchDiscountConfig();
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.detail?.error || errorData.detail || "Failed to update settings";
        NotificationsManager.fromBackend(errorMessage);
      }
    } catch (error) {
      console.error("Error updating discount config:", error);
      NotificationsManager.fromBackend("Failed to update discount configuration");
    } finally {
      setLoading(false);
    }
  };

  const handleAddProvider = () => {
    if (!selectedProvider || !newDiscount) {
      NotificationsManager.fromBackend("Please select a provider and enter discount value");
      return;
    }
    
    const discountValue = parseFloat(newDiscount);
    if (isNaN(discountValue) || discountValue < 0 || discountValue > 1) {
      NotificationsManager.fromBackend("Discount must be between 0 and 1 (0% to 100%)");
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

    setDiscountConfig(prev => ({
      ...prev,
      [providerValue]: discountValue,
    }));
    setSelectedProvider(undefined);
    setNewDiscount("");
    setIsModalVisible(false);
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedProvider(undefined);
    setNewDiscount("");
  };

  const handleRemoveProvider = (provider: string) => {
    setDiscountConfig(prev => {
      const updated = { ...prev };
      delete updated[provider];
      return updated;
    });
  };

  const handleDiscountChange = (provider: string, value: string) => {
    const discountValue = parseFloat(value);
    if (!isNaN(discountValue) && discountValue >= 0 && discountValue <= 1) {
      setDiscountConfig(prev => ({
        ...prev,
        [provider]: discountValue,
      }));
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div style={{ width: "100%" }} className="relative">
      <div className="bg-white rounded-lg shadow w-full">
        <div className="border-b px-6 py-4">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            <div>
              <Title>Cost Tracking Settings</Title>
              <Text className="mt-1 text-sm text-gray-500">
                Configure cost discounts for different LLM providers. Discounts are applied as multipliers.
              </Text>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => setIsModalVisible(true)}
                size="sm"
              >
                + Add Provider Discount
              </Button>
              <Button
                onClick={handleSave}
                loading={loading}
                disabled={loading || isFetching}
                size="sm"
                variant="primary"
              >
                Save Changes
              </Button>
            </div>
          </div>
        </div>

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
                Discounts are applied to provider costs: <code className="bg-gray-200 px-1.5 py-0.5 rounded text-xs">final_cost = base_cost × (1 - discount)</code>
              </Text>
            </div>
            <div>
              <Text className="font-medium text-gray-900 text-sm">Example</Text>
              <Text className="text-xs text-gray-600">
                A 5% discount (0.05) on a $10.00 request results in: $10.00 × (1 - 0.05) = $9.50
              </Text>
            </div>
            <div>
              <Text className="font-medium text-gray-900 text-sm">Valid Range</Text>
              <Text className="text-xs text-gray-600">
                Discount values must be between 0 (0%) and 1 (100%)
              </Text>
            </div>
          </div>
        </div>
      </div>

      <Modal
        title="Add Provider Discount"
        open={isModalVisible}
        onCancel={handleModalCancel}
        footer={null}
        width={600}
      >
        <div className="py-4">
          <Text className="text-sm text-gray-600 mb-4">
            Select a provider and set its discount rate. Discount should be between 0 (0%) and 1 (100%).
          </Text>
          <AddProviderForm
            discountConfig={discountConfig}
            selectedProvider={selectedProvider}
            newDiscount={newDiscount}
            onProviderChange={setSelectedProvider}
            onDiscountChange={setNewDiscount}
            onAddProvider={handleAddProvider}
          />
        </div>
      </Modal>
    </div>
  );
};

export default CostTrackingSettings;
