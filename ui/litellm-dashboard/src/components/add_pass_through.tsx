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
            {/* Route Configuration Section */}
            <Card className="p-6">
              <Title className="text-lg font-semibold text-gray-900 mb-2">Route Configuration</Title>
              <Subtitle className="text-gray-600 mb-6">Configure how requests to your domain will be forwarded to the target API</Subtitle>
              
              <div className="space-y-6">
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Path Prefix
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
                    <div className="text-xs text-gray-500 mt-2">
                      <div className="font-medium mb-1">The path where your API will be accessible. Must start with "/".</div>
                      <div>Example: /bria, /openai, /anthropic</div>
                    </div>
                  }
                >
                  <div className="flex items-center">
                    <span className="mr-1 text-gray-500 font-mono">/</span>
                    <TextInput 
                      placeholder="bria" 
                      value={pathValue.startsWith('/') ? pathValue.slice(1) : pathValue}
                      onChange={(e) => handlePathChange(e.target.value)}
                      className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
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
                    <div className="text-xs text-gray-500 mt-2">
                      <div className="font-medium mb-1">The base URL of the API you want to proxy to. Don't include trailing slash.</div>
                      <div>Example: https://api.openai.com, https://engine.prod.bria-api.com</div>
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
                      Include Subpaths
                      <Tooltip title="Forward all subpaths to the target API (recommended for REST APIs)">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600" />
                      </Tooltip>
                    </span>
                  }
                  name="include_subpath"
                  valuePropName="checked"
                  extra={
                    <div className="text-xs mt-2 p-3 bg-blue-50 rounded-md">
                      <div className="font-medium text-blue-900 mb-2">
                        {includeSubpath ? 'Subpaths enabled:' : 'Subpaths disabled:'}
                      </div>
                      {includeSubpath ? (
                        <div className="text-blue-700">
                          All requests to your path prefix will be forwarded with their full subpath. This allows you to access nested API endpoints like {pathValue}/v1/text-to-image/base/model.
                        </div>
                      ) : (
                        <div className="text-blue-700">
                          Only requests to the exact path prefix will be forwarded without any subpaths.
                        </div>
                      )}
                    </div>
                  }
                >
                  <Switch 
                    checked={includeSubpath}
                    onChange={setIncludeSubpath}
                  />
                </Form.Item>
              </div>
            </Card>

            {/* Headers Section */}
            <Card className="p-6">
              <Title className="text-lg font-semibold text-gray-900 mb-2">Headers</Title>
              <Subtitle className="text-gray-600 mb-6">Add headers that will be sent with every request to the target API</Subtitle>
              
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Authentication Headers
                    <Tooltip title="Authentication and other headers to forward with requests">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="headers"
                rules={[{ required: true, message: 'Please configure the headers' }]}
                extra={
                  <div className="text-xs text-gray-500 mt-2">
                    <div className="font-medium mb-1">Add authentication tokens and other required headers</div>
                    <div>Common examples: auth_token, Authorization, x-api-key</div>
                  </div>
                }
              >
                <KeyValueInput/>
              </Form.Item>
            </Card>

            {/* Route Preview Section */}
            {pathValue && targetValue && (
              <Card className="p-6">
                <Title className="text-lg font-semibold text-gray-900 mb-2">Route Preview</Title>
                <Subtitle className="text-gray-600 mb-6">How your requests will be routed</Subtitle>
                
                <div className="space-y-6">
                  {/* Basic routing */}
                  <div>
                    <div className="text-base font-semibold text-gray-900 mb-4">Basic routing:</div>
                    <div className="flex items-center gap-4">
                      {/* Your endpoint */}
                      <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <div className="text-sm text-gray-600 mb-2">Your endpoint</div>
                        <code className="font-mono text-sm text-gray-900">{getLiteLLMProxyUrl()}</code>
                      </div>
                      
                      {/* Arrow */}
                      <div className="text-gray-400">
                        <RightOutlined className="text-lg" />
                      </div>
                      
                      {/* Forwards to */}
                      <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <div className="text-sm text-gray-600 mb-2">Forwards to</div>
                        <code className="font-mono text-sm text-gray-900">{targetValue}</code>
                      </div>
                    </div>
                  </div>

                  {includeSubpath && (
                    <>
                      {/* With subpaths */}
                      <div>
                        <div className="text-base font-semibold text-gray-900 mb-4">With subpaths:</div>
                        <div className="flex items-center gap-4">
                          {/* Your endpoint + subpath */}
                          <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-4">
                            <div className="text-sm text-gray-600 mb-2">Your endpoint + subpath</div>
                            <code className="font-mono text-sm text-gray-900">
                              {pathValue && `https://your-domain.com${pathValue}`}
                              <span className="text-blue-600">/v1/text-to-image/base/model</span>
                            </code>
                          </div>
                          
                          {/* Arrow */}
                          <div className="text-gray-400">
                            <RightOutlined className="text-lg" />
                          </div>
                          
                          {/* Forwards to with subpath */}
                          <div className="flex-1 bg-gray-50 border border-gray-200 rounded-lg p-4">
                            <div className="text-sm text-gray-600 mb-2">Forwards to</div>
                            <code className="font-mono text-sm text-gray-900">
                              {targetValue}
                              <span className="text-blue-600">/v1/text-to-image/base/model</span>
                            </code>
                          </div>
                        </div>
                        
                        {/* Note */}
                        <div className="mt-4 text-sm text-gray-600">
                          Any path after {pathValue} will be appended to the target URL
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </Card>
            )}

            {/* Billing Section */}
            <Card className="p-6">
              <Title className="text-lg font-semibold text-gray-900 mb-2">Billing</Title>
              <Subtitle className="text-gray-600 mb-6">Optional cost tracking for this endpoint</Subtitle>
              
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
                  <div className="text-xs text-gray-500 mt-2">
                    The cost charged for each request through this endpoint
                  </div>
                }
              >
                <InputNumber 
                  min={0} 
                  step={0.001} 
                  precision={6}
                  placeholder="2.000000"
                  size="large"
                  className="rounded-lg w-full"
                />
              </Form.Item>
            </Card>

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
