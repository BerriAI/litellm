import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  TextInput,
  Button,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Grid,
  Col,
  Subtitle,
} from "@tremor/react";
import { getProxyBaseUrl } from "@/components/networking";
import NotificationsManager from "./molecules/notifications_manager";

interface CostTrackingSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

interface DiscountConfig {
  [provider: string]: number;
}

const CostTrackingSettings: React.FC<CostTrackingSettingsProps> = ({ 
  userID, 
  userRole, 
  accessToken 
}) => {
  const [discountConfig, setDiscountConfig] = useState<DiscountConfig>({});
  const [newProvider, setNewProvider] = useState<string>("");
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
    if (!newProvider || !newDiscount) {
      NotificationsManager.fromBackend("Please enter both provider and discount value");
      return;
    }
    
    const discountValue = parseFloat(newDiscount);
    if (isNaN(discountValue) || discountValue < 0 || discountValue > 1) {
      NotificationsManager.fromBackend("Discount must be between 0 and 1 (0% to 100%)");
      return;
    }

    setDiscountConfig(prev => ({
      ...prev,
      [newProvider.trim()]: discountValue,
    }));
    setNewProvider("");
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

  const hasChanges = Object.keys(discountConfig).length > 0;

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
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableHeaderCell>Provider</TableHeaderCell>
                          <TableHeaderCell>Discount Value</TableHeaderCell>
                          <TableHeaderCell>Percentage</TableHeaderCell>
                          <TableHeaderCell>Actions</TableHeaderCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(discountConfig)
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([provider, discount]) => (
                            <TableRow key={provider}>
                              <TableCell className="font-medium">{provider}</TableCell>
                              <TableCell>
                                <TextInput
                                  value={discount.toString()}
                                  onValueChange={(value) => handleDiscountChange(provider, value)}
                                  placeholder="0.05"
                                  className="w-32"
                                />
                              </TableCell>
                              <TableCell>
                                <span className="text-gray-700 font-medium">
                                  {(discount * 100).toFixed(1)}%
                                </span>
                              </TableCell>
                              <TableCell>
                                <Button
                                  size="xs"
                                  variant="secondary"
                                  color="red"
                                  onClick={() => handleRemoveProvider(provider)}
                                >
                                  Remove
                                </Button>
                              </TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : (
                  <div className="py-8 text-center border border-dashed border-gray-300 rounded-lg">
                    <Text className="text-gray-500">
                      No provider discounts configured. Add your first provider below.
                    </Text>
                  </div>
                )}

                <div className="border-t pt-6 mt-6">
                  <div className="mb-3">
                    <Text className="font-medium text-gray-900">Add Provider Discount</Text>
                    <Text className="text-xs text-gray-500 mt-1">
                      Common providers: vertex_ai, gemini, openai, anthropic, openrouter, bedrock, azure
                    </Text>
                  </div>
                  <Grid numItems={3} className="gap-3">
                    <Col numColSpan={1}>
                      <TextInput
                        placeholder="Provider name"
                        value={newProvider}
                        onValueChange={setNewProvider}
                      />
                    </Col>
                    <Col numColSpan={1}>
                      <TextInput
                        placeholder="Discount (0.05 for 5%)"
                        value={newDiscount}
                        onValueChange={setNewDiscount}
                      />
                    </Col>
                    <Col numColSpan={1}>
                      <Button 
                        onClick={handleAddProvider} 
                        className="w-full"
                        disabled={!newProvider || !newDiscount}
                      >
                        Add Provider
                      </Button>
                    </Col>
                  </Grid>
                </div>
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

