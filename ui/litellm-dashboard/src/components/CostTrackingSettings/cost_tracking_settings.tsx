import React, { useState, useEffect } from "react";
import { Card, Title, Text, Subtitle, Grid, Col, Button } from "@tremor/react";
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
      <div className="mb-6">
        <Title>Cost Tracking Settings</Title>
        <Subtitle>
          Configure cost discounts for different LLM providers. Discounts are applied as multipliers.
        </Subtitle>
      </div>

      <Grid numItems={1} className="gap-6">
        <Col>
          <Card>
            <div className="flex justify-between items-start mb-4">
              <div>
                <Title>Provider Discounts</Title>
                <Text className="mt-1 text-sm text-gray-500">
                  Set custom discount rates per provider (e.g., 0.05 = 5% discount)
                </Text>
              </div>
              <Button
                onClick={handleSave}
                loading={loading}
                disabled={loading || isFetching}
                size="sm"
              >
                Save Changes
              </Button>
            </div>

            {isFetching ? (
              <div className="py-8 text-center">
                <Text className="text-gray-500">Loading configuration...</Text>
              </div>
            ) : (
              <>
                {Object.keys(discountConfig).length > 0 ? (
                  <div className="mt-4">
                    <ProviderDiscountTable
                      discountConfig={discountConfig}
                      onDiscountChange={handleDiscountChange}
                      onRemoveProvider={handleRemoveProvider}
                    />
                  </div>
                ) : (
                  <div className="py-8 text-center border border-dashed border-gray-300 rounded-lg">
                    <Text className="text-gray-500">
                      No provider discounts configured. Add your first provider below.
                    </Text>
                  </div>
                )}

                <AddProviderForm
                  discountConfig={discountConfig}
                  selectedProvider={selectedProvider}
                  newDiscount={newDiscount}
                  onProviderChange={setSelectedProvider}
                  onDiscountChange={setNewDiscount}
                  onAddProvider={handleAddProvider}
                />
              </>
            )}
          </Card>
        </Col>

        <Col>
          <Card>
            <Title>How It Works</Title>
            <div className="mt-4 space-y-3">
              <div>
                <Text className="font-medium text-gray-900">Cost Calculation</Text>
                <Text className="text-sm text-gray-600 mt-1">
                  Discounts are applied to provider costs: <code className="bg-gray-100 px-1 py-0.5 rounded">final_cost = base_cost × (1 - discount)</code>
                </Text>
              </div>
              <div>
                <Text className="font-medium text-gray-900">Example</Text>
                <Text className="text-sm text-gray-600 mt-1">
                  A 5% discount (0.05) on a $10.00 request results in: $10.00 × (1 - 0.05) = $9.50
                </Text>
              </div>
              <div>
                <Text className="font-medium text-gray-900">Valid Range</Text>
                <Text className="text-sm text-gray-600 mt-1">
                  Discount values must be between 0 (0%) and 1 (100%)
                </Text>
              </div>
            </div>
          </Card>
        </Col>
      </Grid>
    </div>
  );
};

export default CostTrackingSettings;
