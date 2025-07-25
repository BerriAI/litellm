import React, { useState } from "react";
import { Card, Form, Button, Input, InputNumber, Select as AntdSelect, Space, Tooltip, Collapse } from "antd";
import { PlusOutlined, DeleteOutlined, InfoCircleOutlined, DownOutlined } from "@ant-design/icons";
import { Text, TextInput } from "@tremor/react";
import { ModelGroup } from "../chat_ui/llm_calls/fetch_models";

const { TextArea } = Input;
const { Panel } = Collapse;

interface Route {
  id: string;
  model: string;
  utterances: string[];
  description: string;
  score_threshold: number;
}

interface RouterConfigBuilderProps {
  modelInfo: ModelGroup[];
  value?: any;
  onChange?: (config: any) => void;
}

const RouterConfigBuilder: React.FC<RouterConfigBuilderProps> = ({
  modelInfo,
  value,
  onChange,
}) => {
  const [routes, setRoutes] = useState<Route[]>(value?.routes || []);
  const [showJsonPreview, setShowJsonPreview] = useState<boolean>(false);

  // Handle adding a new route
  const addRoute = () => {
    const newRoute: Route = {
      id: `route-${Date.now()}`,
      model: "",
      utterances: [],
      description: "",
      score_threshold: 0.5,
    };
    const updatedRoutes = [...routes, newRoute];
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  // Handle removing a route
  const removeRoute = (routeId: string) => {
    const updatedRoutes = routes.filter(route => route.id !== routeId);
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  // Handle updating a route
  const updateRoute = (routeId: string, field: keyof Route, value: any) => {
    const updatedRoutes = routes.map(route => 
      route.id === routeId ? { ...route, [field]: value } : route
    );
    setRoutes(updatedRoutes);
    updateConfig(updatedRoutes);
  };

  // Update the overall configuration
  const updateConfig = (updatedRoutes: Route[]) => {
    const config = {
      routes: updatedRoutes.map(route => ({
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
      .split('\n')
      .map(line => line.trim()) // Only trims leading/trailing whitespace, preserves internal spaces
      .filter(line => line.length > 0);
    updateRoute(routeId, 'utterances', utterancesArray);
  };

  // Prepare model options for dropdowns
  const modelOptions = modelInfo.map(model => ({
    value: model.model_group,
    label: model.model_group,
  }));

  const generateConfig = () => {
    return {
      routes: routes.map(route => ({
        name: route.model,
        utterances: route.utterances,
        description: route.description,
        score_threshold: route.score_threshold,
      })),
    };
  };

  return (
    <div className="w-full">
      {/* Routes Configuration Header */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center gap-2">
          <Text className="text-lg font-semibold">Routes Configuration</Text>
          <Tooltip title="Configure routing logic to automatically select the best model based on user input patterns">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>
        <Button 
          type="primary" 
          icon={<PlusOutlined />} 
          onClick={addRoute}
          className="bg-blue-600 hover:bg-blue-700"
        >
          Add Route
        </Button>
      </div>

      {/* Routes */}
      {routes.length === 0 ? (
        <div className="text-center py-12 text-gray-500 bg-gray-50 rounded-lg border-2 border-dashed border-gray-200 mb-6">
          <Text>No routes configured. Click "Add Route" to get started.</Text>
        </div>
      ) : (
        <div className="space-y-3 mb-6">
          {routes.map((route, index) => (
            <Card 
              key={route.id} 
              className="border border-gray-200 shadow-sm"
              bodyStyle={{ padding: 0 }}
            >
              <Collapse
                ghost
                expandIcon={({ isActive }) => <DownOutlined rotate={isActive ? 180 : 0} />}
                items={[
                  {
                    key: route.id,
                    label: (
                      <div className="flex justify-between items-center py-2">
                        <Text className="font-medium text-base">
                          Route {index + 1}: {route.model || 'Unnamed'}
                        </Text>
                        <Button 
                          type="text" 
                          danger 
                          icon={<DeleteOutlined />}
                          onClick={(e) => {
                            e.stopPropagation();
                            removeRoute(route.id);
                          }}
                          className="mr-2"
                        />
                      </div>
                    ),
                    children: (
                      <div className="px-6 pb-6">
                        {/* Model Selection */}
                        <div className="mb-4">
                          <Text className="text-sm font-medium mb-2 block">Model</Text>
                          <AntdSelect
                            value={route.model}
                            onChange={(value) => updateRoute(route.id, 'model', value)}
                            placeholder="Select model"
                            showSearch
                            style={{ width: '100%' }}
                            options={modelOptions}
                          />
                        </div>

                        {/* Description */}
                        <div className="mb-4">
                          <Text className="text-sm font-medium mb-2 block">Description</Text>
                          <TextArea
                            value={route.description}
                            onChange={(e) => updateRoute(route.id, 'description', e.target.value)}
                            placeholder="Describe when this route should be used..."
                            rows={2}
                          />
                        </div>

                        {/* Score Threshold */}
                        <div className="mb-4">
                          <div className="flex items-center gap-2 mb-2">
                            <Text className="text-sm font-medium">Score Threshold</Text>
                            <Tooltip title="Minimum similarity score to route to this model (0-1)">
                              <InfoCircleOutlined className="text-gray-400" />
                            </Tooltip>
                          </div>
                          <InputNumber
                            value={route.score_threshold}
                            onChange={(value) => updateRoute(route.id, 'score_threshold', value || 0)}
                            min={0}
                            max={1}
                            step={0.1}
                            style={{ width: '100%' }}
                            placeholder="0.5"
                          />
                        </div>

                        {/* Example Utterances */}
                        <div>
                          <div className="flex items-center gap-2 mb-2">
                            <Text className="text-sm font-medium">Example Utterances</Text>
                            <Tooltip title="Training examples for this route. Type an utterance and press Enter to add it.">
                              <InfoCircleOutlined className="text-gray-400" />
                            </Tooltip>
                          </div>
                          <Text className="text-xs text-gray-500 mb-2">Type an utterance and press Enter to add it. You can also paste multiple lines.</Text>
                          <AntdSelect
                            mode="tags"
                            value={route.utterances}
                            onChange={(utterances) => updateRoute(route.id, 'utterances', utterances)}
                            placeholder="Type an utterance and press Enter..."
                            style={{ width: '100%' }}
                            tokenSeparators={['\n']}
                            maxTagCount="responsive"
                            allowClear
                          />
                        </div>
                      </div>
                    ),
                  },
                ]}
              />
            </Card>
          ))}
        </div>
      )}

      {/* JSON Preview */}
      <div className="border-t pt-6">
        <div className="flex justify-between items-center mb-4">
          <Text className="text-lg font-semibold">JSON Preview</Text>
          <Button 
            type="link" 
            onClick={() => setShowJsonPreview(!showJsonPreview)}
            className="text-blue-600 p-0"
          >
            {showJsonPreview ? 'Hide' : 'Show'}
          </Button>
        </div>
        
        {showJsonPreview && (
          <Card className="bg-gray-50">
            <pre className="text-sm overflow-auto max-h-64">
              {JSON.stringify(generateConfig(), null, 2)}
            </pre>
          </Card>
        )}
      </div>
    </div>
  );
};

export default RouterConfigBuilder; 