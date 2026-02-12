import { DeleteOutlined, InfoCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Select as AntdSelect, Button, Card, Collapse, Divider, Empty, Flex, Input, InputNumber, Space, Tooltip, Typography } from "antd";
import React, { useEffect, useState } from "react";
import { ModelGroup } from "../playground/llm_calls/fetch_models";

const { Text } = Typography;

const { TextArea } = Input;

interface Route {
  id: string;
  model: string;
  utterances: string[];
  description: string;
  score_threshold: number;
}

interface SavedRoute {
  id?: string;
  name?: string;
  model?: string;
  utterances?: string[];
  description?: string;
  score_threshold?: number;
}

interface RouterConfig {
  routes?: SavedRoute[];
}

interface RouterConfigBuilderProps {
  modelInfo: ModelGroup[];
  value?: RouterConfig;
  onChange?: (config: any) => void;
}

const RouterConfigBuilder: React.FC<RouterConfigBuilderProps> = ({ modelInfo, value, onChange }) => {
  const [routes, setRoutes] = useState<Route[]>([]);
  const [showJsonPreview, setShowJsonPreview] = useState<boolean>(false);
  const [expandedRoutes, setExpandedRoutes] = useState<string[]>([]);

  // Initialize routes from value prop - preserve existing route IDs to avoid focus loss when parent re-renders
  useEffect(() => {
    const routesFromValue = value?.routes;
    if (routesFromValue) {
      const routeIds: string[] = [];
      setRoutes((prevRoutes) => {
        const initializedRoutes = routesFromValue.map((route: SavedRoute, index: number) => {
          const existingRoute = prevRoutes[index];
          const id = existingRoute?.id || route.id || `route-${index}-${Date.now()}`;
          routeIds.push(id);
          return {
            id,
            model: route.name || route.model || "", // handle both 'name' and 'model' fields
            utterances: route.utterances || [],
            description: route.description || "",
            score_threshold: route.score_threshold ?? 0.5,
          };
        });
        return initializedRoutes;
      });
      setExpandedRoutes(routeIds);
    } else {
      setRoutes([]);
      setExpandedRoutes([]);
    }
  }, [value]);

  // Handle adding a new route
  const addRoute = () => {
    const newRouteId = `route-${Date.now()}`;
    const newRoute: Route = {
      id: newRouteId,
      model: "",
      utterances: [],
      description: "",
      score_threshold: 0.5,
    };
    const updatedRoutes = [...routes, newRoute];
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    // Automatically expand the new route
    setExpandedRoutes((prev) => [...prev, newRouteId]);
  };

  // Handle removing a route
  const removeRoute = (routeId: string) => {
    const updatedRoutes = routes.filter((route) => route.id !== routeId);
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
    // Remove from expanded routes as well
    setExpandedRoutes((prev) => prev.filter((id) => id !== routeId));
  };

  // Handle updating a route
  const updateRoute = (routeId: string, field: keyof Route, value: any) => {
    const updatedRoutes = routes.map((route) => (route.id === routeId ? { ...route, [field]: value } : route));
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  // Update the overall configuration
  const updateConfig = (updatedRoutes: Route[]) => {
    const config = {
      routes: updatedRoutes.map((route) => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    };
    onChange?.(config);
  };

  // Handle utterances change (convert textarea string to array)
  const handleUtterancesChange = (routeId: string, utterancesText: string) => {
    const utterancesArray = utterancesText
      .split("\n")
      .map((line) => line.trim()) // Only trims leading/trailing whitespace, preserves internal spaces
      .filter((line) => line.length > 0);
    updateRoute(routeId, "utterances", utterancesArray);
  };

  // Prepare model options for dropdowns
  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  const generateConfig = () => {
    return {
      routes: routes.map((route) => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    };
  };

  return (
    <div className="w-full max-w-none">
      <Flex justify="space-between" align="center" gap="middle" style={{ width: "100%", marginBottom: 24 }}>
        <Space align="center">
          <Typography.Title level={4} style={{ margin: 0 }}>Routes Configuration</Typography.Title>
          <Tooltip title="Configure routing logic to automatically select the best model based on user input patterns">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </Space>
        <Button type="primary" icon={<PlusOutlined />} onClick={addRoute} className="bg-blue-600 hover:bg-blue-700">
          Add Route
        </Button>
      </Flex>

      {/* Routes */}
      {routes.length === 0 ? (
        <Card>
          <Empty description="No routes configured. Click &quot;Add Route&quot; to get started." />
        </Card>
      ) : (
        <Collapse
          activeKey={expandedRoutes}
          onChange={(keys) => setExpandedRoutes(Array.isArray(keys) ? keys : [keys].filter(Boolean))}
          style={{ width: "100%" }}
          items={routes.map((route, index) => ({
            key: route.id,
            label: (
              <Text style={{ fontSize: 16 }}>
                Route {index + 1}: {route.model || "Unnamed"}
              </Text>
            ),
            extra: (
              <Button
                type="text"
                danger
                size="small"
                icon={<DeleteOutlined />}
                onClick={(e) => {
                  e.stopPropagation();
                  removeRoute(route.id);
                }}
              />
            ),
            children: (
              <Card key={route.id}>
                {/* Model Selection */}
                <div className="mb-4 w-full">
                  <Text className="text-sm font-medium mb-2 block">Model</Text>
                  <AntdSelect
                    value={route.model}
                    onChange={(value) => updateRoute(route.id, "model", value)}
                    placeholder="Select model"
                    showSearch
                    style={{ width: "100%" }}
                    options={modelOptions}
                  />
                </div>

                {/* Description */}
                <div className="mb-4 w-full">
                  <Text className="text-sm font-medium mb-2 block">Description</Text>
                  <TextArea
                    value={route.description}
                    onChange={(e) => updateRoute(route.id, "description", e.target.value)}
                    placeholder="Describe when this route should be used..."
                    rows={2}
                    style={{ width: "100%" }}
                  />
                </div>

                {/* Score Threshold */}
                <div className="mb-4 w-full">
                  <div className="flex items-center gap-2 mb-2">
                    <Text className="text-sm font-medium">Score Threshold</Text>
                    <Tooltip title="Minimum similarity score to route to this model (0-1)">
                      <InfoCircleOutlined className="text-gray-400" />
                    </Tooltip>
                  </div>
                  <InputNumber
                    value={route.score_threshold}
                    onChange={(value) => updateRoute(route.id, "score_threshold", value || 0)}
                    min={0}
                    max={1}
                    step={0.1}
                    style={{ width: "100%" }}
                    placeholder="0.5"
                  />
                </div>

                {/* Example Utterances */}
                <div className="w-full">
                  <div className="flex items-center gap-2 mb-2">
                    <Text className="text-sm font-medium">Example Utterances</Text>
                    <Tooltip title="Training examples for this route. Type an utterance and press Enter to add it.">
                      <InfoCircleOutlined className="text-gray-400" />
                    </Tooltip>
                  </div>
                  <Text className="text-xs text-gray-500 mb-2">
                    Type an utterance and press Enter to add it. You can also paste multiple lines.
                  </Text>
                  <AntdSelect
                    mode="tags"
                    value={route.utterances}
                    onChange={(utterances) => updateRoute(route.id, "utterances", utterances)}
                    placeholder="Type an utterance and press Enter..."
                    style={{ width: "100%" }}
                    tokenSeparators={["\n"]}
                    maxTagCount="responsive"
                    allowClear
                  />
                </div>
              </Card>
            ),
          }))}
        />
      )}

      {/* JSON Preview */}
      <Divider />
      <div className="flex justify-between items-center mb-4 w-full">
        <Text className="text-lg font-semibold">JSON Preview</Text>
        <Button type="link" onClick={() => setShowJsonPreview(!showJsonPreview)} className="text-blue-600 p-0">
          {showJsonPreview ? "Hide" : "Show"}
        </Button>
      </div>

      {showJsonPreview && (
        <Card className="bg-gray-50 w-full">
          <pre className="text-sm overflow-auto max-h-64 w-full">{JSON.stringify(generateConfig(), null, 2)}</pre>
        </Card>
      )}

    </div>
  );
};

export default RouterConfigBuilder;
