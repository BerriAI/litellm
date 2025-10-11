/**
 * Modal to add fallbacks to the proxy router config
 */

import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { SearchSelect, SearchSelectItem } from "@tremor/react";
import { setCallbacksCall } from "./networking";
import { Modal, Form } from "antd";
import { fetchAvailableModels, ModelGroup } from "./chat_ui/llm_calls/fetch_models";
import NotificationManager from "./molecules/notifications_manager";

interface AddFallbacksProps {
  models?: string[];
  accessToken: string;
  routerSettings: { [key: string]: any };
  setRouterSettings: React.Dispatch<React.SetStateAction<{ [key: string]: any }>>;
}

const AddFallbacks: React.FC<AddFallbacksProps> = ({ models, accessToken, routerSettings, setRouterSettings }) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [selectedFallbacks, setSelectedFallbacks] = useState<string[]>([]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        console.log("Fetched models for fallbacks:", uniqueModels);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for fallbacks:", error);
      }
    };
    loadModels();
  }, [accessToken]);
  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedFallbacks([]);
    setSelectedModel("");
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
    setSelectedFallbacks([]);
    setSelectedModel("");
  };

  const updateFallbacks = (formValues: Record<string, any>) => {
    // Print the received value
    console.log(formValues);

    // Extract model_name and models from formValues
    const { model_name, models } = formValues;

    // Create new fallback
    const newFallback = { [model_name]: models };

    // Get current fallbacks, or an empty array if it's null
    const currentFallbacks = routerSettings.fallbacks || [];

    // Add new fallback to the current fallbacks
    const updatedFallbacks = [...currentFallbacks, newFallback];

    // Create a new routerSettings object with updated fallbacks
    const updatedRouterSettings = { ...routerSettings, fallbacks: updatedFallbacks };

    // Print updated routerSettings
    console.log(updatedRouterSettings);

    const payload = {
      router_settings: updatedRouterSettings,
    };

    try {
      setCallbacksCall(accessToken, payload);
      // Update routerSettings state
      setRouterSettings(updatedRouterSettings);
    } catch (error) {
      NotificationManager.fromBackend("Failed to update router settings: " + error);
    }

    NotificationManager.success("router settings updated successfully");

    setIsModalVisible(false);
    form.resetFields();
    setSelectedFallbacks([]);
    setSelectedModel("");
  };

  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)} icon={() => <span className="mr-1">+</span>}>
        Add Fallbacks
      </Button>
      <Modal
        title={
          <div className="pb-4 border-b border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900">Add Fallbacks</h2>
          </div>
        }
        open={isModalVisible}
        width={900}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
        className="top-8"
        styles={{
          body: { padding: "24px" },
          header: { padding: "24px 24px 0 24px", border: "none" },
        }}
      >
        <div className="mt-6">
          <div className="mb-6">
            <p className="text-gray-600">
              Configure fallback models to improve reliability. When the primary model fails or is unavailable, requests
              will automatically route to the specified fallback models in order.
            </p>
          </div>

          <Form form={form} onFinish={updateFallbacks} layout="vertical" className="space-y-6">
            <div className="grid grid-cols-1 gap-6">
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700">
                    Primary Model <span className="text-red-500">*</span>
                  </span>
                }
                name="model_name"
                rules={[{ required: true, message: "Please select the primary model that needs fallbacks" }]}
                className="!mb-0"
              >
                <SearchSelect
                  placeholder="Select the model that needs fallback protection"
                  value={selectedModel}
                  onValueChange={(value: string) => {
                    setSelectedModel(value);
                    // Remove the selected model from fallbacks if it was selected
                    const updatedFallbacks = selectedFallbacks.filter((model) => model !== value);
                    setSelectedFallbacks(updatedFallbacks);
                    form.setFieldValue("models", updatedFallbacks);
                    form.setFieldValue("model_name", value);
                  }}
                >
                  {Array.from(new Set(modelInfo.map((option) => option.model_group))).map(
                    (model: string, index: number) => (
                      <SearchSelectItem key={index} value={model}>
                        {model}
                      </SearchSelectItem>
                    ),
                  )}
                </SearchSelect>
                <p className="text-sm text-gray-500 mt-1">This is the primary model that users will request</p>
              </Form.Item>

              <div className="border-t border-gray-200 my-6"></div>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700">
                    Fallback Models (select multiple) <span className="text-red-500">*</span>
                  </span>
                }
                name="models"
                rules={[{ required: true, message: "Please select at least one fallback model" }]}
                className="!mb-0"
              >
                <div className="space-y-3">
                  {/* Show selected models in order */}
                  {selectedFallbacks.length > 0 && (
                    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
                      <p className="text-sm font-medium text-gray-700 mb-2">Fallback Order:</p>
                      <div className="flex flex-wrap gap-2">
                        {selectedFallbacks.map((model, index) => (
                          <div
                            key={model}
                            className="flex items-center bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm"
                          >
                            <span className="font-medium mr-2">{index + 1}.</span>
                            <span>{model}</span>
                            <button
                              type="button"
                              onClick={() => {
                                const newFallbacks = selectedFallbacks.filter((m) => m !== model);
                                setSelectedFallbacks(newFallbacks);
                                form.setFieldValue("models", newFallbacks);
                              }}
                              className="ml-2 text-blue-600 hover:text-blue-800"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Model selector */}
                  <SearchSelect
                    placeholder="Add a fallback model"
                    value=""
                    onValueChange={(value: string) => {
                      if (value && !selectedFallbacks.includes(value)) {
                        const newFallbacks = [...selectedFallbacks, value];
                        setSelectedFallbacks(newFallbacks);
                        form.setFieldValue("models", newFallbacks);
                      }
                    }}
                  >
                    {Array.from(new Set(modelInfo.map((option) => option.model_group)))
                      .filter((data: string) => data !== selectedModel && !selectedFallbacks.includes(data))
                      .sort()
                      .map((model: string) => (
                        <SearchSelectItem key={model} value={model}>
                          {model}
                        </SearchSelectItem>
                      ))}
                  </SearchSelect>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  <strong>Order matters:</strong> Models will be tried in the order shown above (1st, 2nd, 3rd, etc.)
                </p>
              </Form.Item>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
              <Button variant="secondary" onClick={handleCancel}>
                Cancel
              </Button>
              <Button variant="primary" type="submit">
                Add Fallbacks
              </Button>
            </div>
          </Form>
        </div>
      </Modal>
    </div>
  );
};

export default AddFallbacks;
