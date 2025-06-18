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
} from "antd";
import { InfoCircleOutlined, ApiOutlined } from "@ant-design/icons";
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
  const handleCancel = () => {
    form.resetFields();
    setIsModalVisible(false);
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
        "input_cost_per_request": formValues["input_cost_per_request"] || 0
      }
      
      await createPassThroughEndpoint(accessToken, formValues);
      
      const updatedPassThroughSettings = [...passThroughItems, newPassThroughItem]
      setPassThroughItems(updatedPassThroughSettings)
      
      message.success("Pass-through endpoint created successfully");
      form.resetFields();
      setIsModalVisible(false);
    } catch (error) {
      message.error("Error creating pass-through endpoint: " + error, 20);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div>
      <Button 
        className="mx-auto mb-4" 
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
          <Form
            form={form}
            onFinish={addPassThrough}
            layout="vertical"
            className="space-y-6"
          >
            <div className="grid grid-cols-1 gap-6">
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Path
                    <Tooltip title="The route to be added to the LiteLLM Proxy Server (e.g., /my-endpoint)">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="path"
                rules={[{ required: true, message: 'Please enter the endpoint path' }]}
              >
                <TextInput 
                  placeholder="e.g., /my-endpoint" 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Target URL
                    <Tooltip title="The URL to which requests for this path should be forwarded">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="target"
                rules={[
                  { required: true, message: 'Please enter the target URL' },
                  { type: 'url', message: 'Please enter a valid URL' }
                ]}
              >
                <TextInput 
                  placeholder="https://your-service.com/api" 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Headers
                    <Tooltip title="Key-value pairs of headers to be forwarded with the request">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="headers"
                rules={[{ required: true, message: 'Please configure the headers' }]}
              >
                <KeyValueInput/>
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Include Subpath
                    <Tooltip title="If enabled, requests to subpaths will also be forwarded to the target endpoint">
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  </span>
                }
                name="include_subpath"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Cost Per Request (USD)
                    <Tooltip title="The cost in USD per request to the target endpoint">
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  </span>
                }
                name="input_cost_per_request"
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
