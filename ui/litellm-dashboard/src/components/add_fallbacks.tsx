/**
 * Modal to add fallbacks to the proxy router config
 */



import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { Select, SelectItem, MultiSelect, MultiSelectItem, Card, Metric, Text, Title, Subtitle, Accordion, AccordionHeader, AccordionBody, } from "@tremor/react";
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { setCallbacksCall } from "./networking";
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

const { Option } = Select2;

interface AddFallbacksProps {
  models: string[] | undefined; 
  accessToken: string;
  routerSettings: { [key: string]: any; }
  setRouterSettings: React.Dispatch<React.SetStateAction<{ [key: string]: any }>>;
}

const AddFallbacks: React.FC<AddFallbacksProps> = ({
    models, 
    accessToken,
    routerSettings,
    setRouterSettings
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
        router_settings: updatedRouterSettings
    };

    try {
        setCallbacksCall(accessToken, payload);
        // Update routerSettings state
        setRouterSettings(updatedRouterSettings);
    } catch (error) {
        message.error("Failed to update router settings: " + error, 20);
    }

    message.success("router settings updated successfully");

    setIsModalVisible(false)
    form.resetFields();
  };


  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
        + Add Fallbacks
      </Button>
      <Modal
        title="Add Fallbacks"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          onFinish={updateFallbacks}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
            <>
              <Form.Item 
                label="Public Model Name" 
                name="model_name"
                rules={[{ required: true, message: 'Set the model to fallback for' }]}
                help="required"
              >
                <Select defaultValue={selectedModel}>
                {models && models.map((model: string, index) => (
                    <SelectItem
                        key={index}
                        value={model}
                        onClick={() => setSelectedModel(model)}
                    >
                        {model}
                    </SelectItem>
                    ))
                }
                </Select>
              </Form.Item>

              <Form.Item 
                label="Fallback Models" 
                name="models"
                rules={[{ required: true, message: 'Please select a model' }]}
                help="required"
              >
                <MultiSelect value={models}>
                    {models &&
                      models.filter(data => data != selectedModel).map((model: string) => (
                        (
                          <MultiSelectItem key={model} value={model}>
                            {model}
                          </MultiSelectItem>
                        )
                      ))}

                </MultiSelect>
              </Form.Item>
            </>
          
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Add Fallbacks</Button2>
          </div>
        </Form>
      </Modal>

    </div>
  );
};

export default AddFallbacks;
