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
import { InfoCircleOutlined, ApiOutlined, ExclamationCircleOutlined, CheckCircleOutlined, CopyOutlined } from "@ant-design/icons";
import { keyCreateCall, slackBudgetAlertsHealthCheck, modelAvailableCall } from "./networking";
import { list } from "postcss";
import KeyValueInput from "./key_value_input";
import { passThroughItem } from "./pass_through_settings";
import RoutePreview from "./route_preview";
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
            <Card className="p-5">
              <Title className="text-lg font-semibold text-gray-900 mb-2">Route Configuration</Title>
              <Subtitle className="text-gray-600 mb-5">Configure how requests to your domain will be forwarded to the target API</Subtitle>
              
              <div className="space-y-5">
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Path Prefix <span className="text-red-500">*</span>
                    </span>
                  }
                  name="path"
                  rules={[
                    { required: true, message: '' },
                    { pattern: /^\//, message: '' }
                  ]}
                  extra={
                    <div className="text-xs text-gray-500 mt-1">
                      Example: /bria, /openai, /anthropic
                    </div>
                  }
                  className="mb-4"
                >
                  <div className="flex items-center">
                    <span className="mr-2 text-gray-500 font-mono text-base">/</span>
                    <TextInput 
                      placeholder="bria" 
                      value={pathValue.startsWith('/') ? pathValue.slice(1) : pathValue}
                      onChange={(e) => handlePathChange(e.target.value)}
                      className="flex-1"
                    />
                  </div>
                </Form.Item>

                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Target URL <span className="text-red-500">*</span>
                    </span>
                  }
                  name="target"
                  rules={[
                    { required: true, message: '' },
                    { type: 'url', message: '' }
                  ]}
                  extra={
                    <div className="text-xs text-gray-500 mt-1">
                      Example: https://api.openai.com, https://engine.prod.bria-api.com
                    </div>
                  }
                  className="mb-4"
                >
                  <TextInput 
                    placeholder="https://engine.prod.bria-api.com" 
                    value={targetValue}
                    onChange={(e) => setTargetValue(e.target.value)}
                  />
                </Form.Item>

                <div className="flex items-center justify-between py-3">
                  <div>
                    <div className="text-sm font-medium text-gray-700">Include Subpaths</div>
                    <div className="text-xs text-gray-500 mt-0.5">Forward all subpaths to the target API (recommended for REST APIs)</div>
                  </div>
                  <Form.Item
                    name="include_subpath"
                    valuePropName="checked"
                    className="mb-0"
                  >
                    <Switch 
                      checked={includeSubpath}
                      onChange={setIncludeSubpath}
                    />
                  </Form.Item>
                </div>
              </div>
            </Card>

            {/* Route Preview Section */}
            <RoutePreview 
              pathValue={pathValue}
              targetValue={targetValue}
              includeSubpath={includeSubpath}
            />

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
