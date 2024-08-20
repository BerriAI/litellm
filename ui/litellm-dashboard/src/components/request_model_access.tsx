"use client";

import React, { useState, useEffect, useRef } from "react";
import { Modal, Form, Input, Select, InputNumber, message } from "antd";
import { Button } from "@tremor/react";
import { userRequestModelCall } from "./networking";

const { Option } = Select;

interface RequestAccessProps {
  userModels: string[];
  accessToken: string;
  userID: string;
}
const isLocal = process.env.NODE_ENV === "development";
const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
if (isLocal != true) {
  console.log = function() {};
}
function onRequestAccess(formData: Record<string, any>): void {
    // This function does nothing for now
  }

const RequestAccess: React.FC<RequestAccessProps> = ({ userModels, accessToken, userID }) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleRequestAccess = async (formValues: Record<string, any>) => {
    try {
      message.info("Requesting access");
      // Extract form values
        const { selectedModel, accessReason } = formValues;

        // Call userRequestModelCall
        const response = await userRequestModelCall(
        accessToken,  // You need to have accessToken available
        selectedModel,
        userID,  // You need to have UserID available
        accessReason
        );
    
      onRequestAccess(formValues);
      setIsModalVisible(true);
    } catch (error) {
      console.error("Error requesting access:", error);
    }
  };

  return (
    <div>
      <Button size="xs" onClick={() => setIsModalVisible(true)}>
        Request Access
      </Button>
      <Modal
        title="Request Access"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          onFinish={handleRequestAccess}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <Form.Item label="Select Model" name="selectedModel">
            <Select placeholder="Select model" style={{ width: '100%' }}>
              {userModels.map((model) => (
                <Option key={model} value={model}>
                  {model}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="Reason for Access" name="accessReason">
            <Input.TextArea rows={4} placeholder="Enter reason for access" />
          </Form.Item>
          <div style={{ textAlign: 'right', marginTop: '10px' }}>
            <Button>Request Access</Button>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default RequestAccess;
