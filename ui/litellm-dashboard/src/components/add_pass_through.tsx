/**
 * Modal to add fallbacks to the proxy router config
 */



import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
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
} from "antd";
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
  const [selectedModel, setSelectedModel] = useState("");
  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const addPassThrough = (formValues: Record<string, any>) => {
    // Print the received value
    console.log(formValues);

    // // Extract model_name and models from formValues
    // const { model_name, models } = formValues;

    // // Create new fallback
    // const newFallback = { [model_name]: models };

    // // Get current fallbacks, or an empty array if it's null
    // const currentFallbacks = routerSettings.fallbacks || [];

    // // Add new fallback to the current fallbacks
    // const updatedFallbacks = [...currentFallbacks, newFallback];

    // // Create a new routerSettings object with updated fallbacks
    // const updatedRouterSettings = { ...routerSettings, fallbacks: updatedFallbacks };

    const newPassThroughItem: passThroughItem = {
        "headers": formValues["headers"],
        "path": formValues["path"],
        "target": formValues["target"]
    }
    const updatedPassThroughSettings = [...passThroughItems, newPassThroughItem]


    try {
        createPassThroughEndpoint(accessToken, formValues);
        setPassThroughItems(updatedPassThroughSettings)
    } catch (error) {
        message.error("Failed to update router settings: " + error, 20);
    }

    message.success("Pass through endpoint successfully added");

    setIsModalVisible(false)
    form.resetFields();
  };


  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
        + Add Pass-Through Endpoint
      </Button>
      <Modal
        title="Add Pass-Through Endpoint"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          onFinish={addPassThrough}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
            <>
              <Form.Item 
                label="Path" 
                name="path"
                rules={[{ required: true, message: 'The route to be added to the LiteLLM Proxy Server.' }]}
                help="required"
              >
                <TextInput/>
              </Form.Item>

              <Form.Item 
                label="Target" 
                name="target"
                rules={[{ required: true, message: 'The URL to which requests for this path should be forwarded.' }]}
                help="required"
              >
                <TextInput/>
              </Form.Item>
              <Form.Item 
                label="Headers" 
                name="headers"
                rules={[{ required: true, message: 'Key-value pairs of headers to be forwarded with the request. You can set any key value pair here and it will be forwarded to your target endpoint' }]}
                help="required"
              >
                <KeyValueInput/>
              </Form.Item>
            </>
          
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Add Pass-Through Endpoint</Button2>
          </div>
        </Form>
      </Modal>

    </div>
  );
};

export default AddPassThroughEndpoint;
