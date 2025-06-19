/**
 * Modal to add fallbacks to the proxy router config
 */



import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col, Switch } from "@tremor/react";
import { Select, SelectItem, MultiSelect, MultiSelectItem, Card, Metric, Text, Title, Subtitle, Accordion, AccordionHeader, AccordionBody, } from "@tremor/react";
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { createPassThroughEndpoint } from "./networking";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  InputNumber,
  Select as Select2,
  message,
  Tooltip,
  Alert,
  Divider,
  Collapse,
} from "antd";
import { InfoCircleOutlined, ApiOutlined, ExclamationCircleOutlined, CheckCircleOutlined, CopyOutlined, RightOutlined } from "@ant-design/icons";
import { keyCreateCall, slackBudgetAlertsHealthCheck, modelAvailableCall } from "./networking";
import { list } from "postcss";
import KeyValueInput from "./key_value_input";
import { passThroughItem } from "./pass_through_settings";
const { Option } = Select2;

interface AddFallbacksProps {
//   models: string[] | undefined; 
  accessToken: string;
  passThroughItems: passThroughItem[];
  setPassThroughItems: React.Dispatch<React.SetStateAction<passThroughItem[]>>;
}

const AddPassThroughEndpoint: React.FC<AddFallbacksProps> = ({
    accessToken, setPassThroughItems, passThroughItems
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [pathValue, setPathValue] = useState("");
  const [targetValue, setTargetValue] = useState("");
  const [includeSubpath, setIncludeSubpath] = useState(true);

  const handleCancel = () => {
    form.resetFields();
    setPathValue("");
    setTargetValue("");
    setIncludeSubpath(true);
    setIsModalVisible(false);
  };

  const handlePathChange = (value: string) => {
    // Auto-add leading slash if missing
    let formattedPath = value;
    if (value && !value.startsWith('/')) {
      formattedPath = '/' + value;
    }
    setPathValue(formattedPath);
    form.setFieldsValue({ path: formattedPath });
  };

  const addPassThrough = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      console.log(`formValues: ${JSON.stringify(formValues)}`);

      const newPassThroughItem: passThroughItem = {
        "headers": formValues["headers"],
        "path": formValues["path"],
        "target": formValues["target"],
        "include_subpath": formValues["include_subpath"] || false,
        "cost_per_request": formValues["cost_per_request"] || 0
      }
      
      await createPassThroughEndpoint(accessToken, formValues);
      
      const updatedPassThroughSettings = [...passThroughItems, newPassThroughItem]
      setPassThroughItems(updatedPassThroughSettings)
      
      message.success("Pass-through endpoint created successfully");
      form.resetFields();
      setPathValue("");
      setTargetValue("");
      setIncludeSubpath(true);
      setIsModalVisible(false);
    } catch (error) {
      message.error("Error creating pass-through endpoint: " + error, 20);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success('Copied to clipboard!');
  };

  const getLiteLLMProxyUrl = () => {
    return pathValue ? `https://your-domain.com${pathValue}` : '';
  };

  const getSubpathExampleUrl = () => {
    return pathValue ? `https://your-domain.com${pathValue}/v1/text-to-image/base/model` : '';
  };

  const getTargetSubpathUrl = () => {
    return targetValue ? `${targetValue}/v1/text-to-image/base/model` : '';
  };

  return (
    <div>
      <Button 
        className="mx-auto mb-4 mt-4" 
        onClick={() => setIsModalVisible(true)}
      >
        + Add Pass-Through Endpoint
      </Button>
      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
            <ApiOutlined className="text-xl text-blue-500" />
            <h2 className="text-xl font-semibold text-gray-900">Add Pass-Through Endpoint</h2>
          </div>
        }
        open={isModalVisible}
        width={1000}
        onCancel={handleCancel}
        footer={null}
        className="top-8"
        styles={{
          body: { padding: '24px' },
          header: { padding: '24px 24px 0 24px', border: 'none' },
        }}
      >
        <div className="mt-6">
          <Alert
            message="What is a Pass-Through Endpoint?"
            description="Route requests from your LiteLLM proxy to any external API. Perfect for custom models, image generation APIs, or any service you want to proxy through LiteLLM."
            type="info"
            showIcon
            className="mb-6"
          />

          <Form
            form={form}
            onFinish={addPassThrough}
            layout="vertical"
            className="space-y-6"
            initialValues={{ include_subpath: true }}
          >
            <div className="grid grid-cols-1 gap-6">
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Path
                    <Tooltip title="The route on your LiteLLM proxy. Must start with '/'. Example: /bria for Bria API">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="path"
                rules={[
                  { required: true, message: 'Please enter the endpoint path' },
                  { pattern: /^\//, message: 'Path must start with /' }
                ]}
                extra={
                  <div className="text-xs text-gray-500 mt-1">
                    <div>✓ Good: /bria, /openai, /custom-api</div>
                    <div>✗ Bad: bria, openai (missing leading slash)</div>
                  </div>
                }
              >
                <TextInput 
                  placeholder="/bria" 
                  value={pathValue}
                  onChange={(e) => handlePathChange(e.target.value)}
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Target URL
                    <Tooltip title="The base URL of the API you want to forward requests to">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="target"
                rules={[
                  { required: true, message: 'Please enter the target URL' },
                  { type: 'url', message: 'Please enter a valid URL' }
                ]}
                extra={
                  <div className="text-xs text-gray-500 mt-1">
                    <div>Examples:</div>
                    <div>• https://engine.prod.bria-api.com</div>
                    <div>• https://api.openai.com</div>
                    <div>• https://your-custom-api.example.com</div>
                  </div>
                }
              >
                <TextInput 
                  placeholder="https://engine.prod.bria-api.com" 
                  value={targetValue}
                  onChange={(e) => setTargetValue(e.target.value)}
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Headers
                    <Tooltip title="Authentication and other headers to forward with requests">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="headers"
                rules={[{ required: true, message: 'Please configure the headers' }]}
                extra={
                  <div className="text-xs text-gray-500 mt-1">
                    Common examples: auth_token, Authorization, x-api-key
                  </div>
                }
              >
                <KeyValueInput/>
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Include Subpath
                    <Tooltip title="Forward the full path including subpaths to the target URL">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600" />
                    </Tooltip>
                  </span>
                }
                name="include_subpath"
                valuePropName="checked"
                extra={
                  <div className="text-xs mt-2">
                    <div className="mb-2 font-medium text-gray-600">Choose based on your API structure:</div>
                    <div className="space-y-1">
                      <div className="flex items-center text-green-600">
                        <CheckCircleOutlined className="mr-2" />
                        <span className="font-medium">Enable</span> if you need routes like: /bria/v1/text-to-image/base/model
                      </div>
                      <div className="flex items-center text-orange-600">
                        <ExclamationCircleOutlined className="mr-2" />
                        <span className="font-medium">Disable</span> if you only need the exact path: /bria
                      </div>
                    </div>
                  </div>
                }
              >
                <Switch 
                  checked={includeSubpath}
                  onChange={setIncludeSubpath}
                />
              </Form.Item>

              {pathValue && targetValue && (
                <Collapse
                  items={[
                    {
                      key: '1',
                      label: (
                        <div className="flex items-center">
                          <RightOutlined className="mr-2" />
                          <span className="font-medium">Route Preview</span>
                        </div>
                      ),
                      children: (
                        <div className="space-y-6">
                          {/* Your API Endpoint */}
                          <div>
                            <div className="text-sm font-medium text-gray-700 mb-3">Your API Endpoint:</div>
                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium">
                                  GET/POST
                                </span>
                                <code className="font-mono text-sm">{getLiteLLMProxyUrl()}</code>
                              </div>
                                                             <Button 
                                 size="xs"
                                 variant="secondary"
                                 onClick={() => copyToClipboard(getLiteLLMProxyUrl())}
                               >
                                 Copy
                               </Button>
                            </div>
                          </div>

                          {/* Arrow */}
                          <div className="flex justify-center">
                            <div className="text-gray-400">
                              <RightOutlined className="text-xl" />
                            </div>
                          </div>

                          {/* Forwards to */}
                          <div>
                            <div className="text-sm font-medium text-gray-700 mb-3">Forwards to:</div>
                            <div className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                              <code className="font-mono text-sm">{targetValue}</code>
                            </div>
                          </div>

                          {includeSubpath && (
                            <>
                              {/* With Subpaths Example */}
                              <div>
                                <div className="text-sm font-medium text-blue-600 mb-3">With Subpaths (Example):</div>
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex items-center justify-between">
                                  <div className="flex items-center space-x-3">
                                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs font-medium">
                                      POST
                                    </span>
                                    <code className="font-mono text-sm">{getSubpathExampleUrl()}</code>
                                  </div>
                                                                     <Button 
                                     size="xs"
                                     variant="secondary"
                                     onClick={() => copyToClipboard(getSubpathExampleUrl())}
                                   >
                                     Copy
                                   </Button>
                                </div>
                              </div>

                              {/* Arrow */}
                              <div className="flex justify-center">
                                <div className="text-gray-400">
                                  <RightOutlined className="text-xl" />
                                </div>
                              </div>

                              {/* Also forwards to */}
                              <div>
                                <div className="text-sm font-medium text-blue-600 mb-3">Also forwards to:</div>
                                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                                  <code className="font-mono text-sm">{getTargetSubpathUrl()}</code>
                                </div>
                              </div>

                              {/* Note */}
                              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                                <div className="text-xs text-blue-700">
                                  <strong>Note:</strong> Any path after {pathValue} will be automatically appended to the target URL
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      ),
                    },
                  ]}
                  defaultActiveKey={['1']}
                  className="border-0"
                  ghost
                />
              )}

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Cost Per Request (USD)
                    <Tooltip title="Optional: Track costs for requests to this endpoint">
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  </span>
                }
                name="cost_per_request"
                extra={
                  <div className="text-xs text-gray-500 mt-1">
                    Leave as 0 if you don't want to track costs for this endpoint
                  </div>
                }
              >
                <InputNumber 
                  min={0} 
                  step={0.001} 
                  precision={6}
                  placeholder="0.000000"
                  size="large"
                  className="rounded-lg w-full"
                />
              </Form.Item>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
              <Button 
                variant="secondary"
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button 
                variant="primary"
                loading={isLoading}
              >
                {isLoading ? 'Creating...' : 'Add Pass-Through Endpoint'}
              </Button>
            </div>
          </Form>
        </div>
      </Modal>

    </div>
  );
};

export default AddPassThroughEndpoint;
