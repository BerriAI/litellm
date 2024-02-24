"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { Card, Metric, Text } from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  message,
} from "antd";
import { keyCreateCall } from "./networking";

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  userRole: string | null;
  accessToken: string;
  data: any[] | null;
  userModels: string[];
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  userRole,
  accessToken,
  data,
  userModels,
  setData,
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);

  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    form.resetFields();
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      message.info("Making API Call");
      setIsModalVisible(true);
      const response = await keyCreateCall(accessToken, userID, formValues);
      setData((prevData) => (prevData ? [...prevData, response] : [response])); // Check if prevData is null
      setApiKey(response["key"]);
      message.success("API Key Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
        + Create New Key
      </Button>
      <Modal
        title="Create Key"
        visible={isModalVisible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Form
          form={form}
          onFinish={handleCreate}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          {userRole === "App Owner" || userRole === "Admin" ? (
            <>
              <Form.Item label="Key Name" name="key_alias">
                <Input />
              </Form.Item>
              <Form.Item label="Team ID" name="team_id">
                <Input placeholder="ai_team" />
              </Form.Item>
              <Form.Item label="Models" name="models">
                <Select
                  mode="multiple"
                  placeholder="Select models"
                  style={{ width: "100%" }}
                >
                  {userModels.map((model) => (
                    <Option key={model} value={model}>
                      {model}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
              <Form.Item label="Max Budget (USD)" name="max_budget">
                <InputNumber step={0.01} precision={2} width={200} />
              </Form.Item>
              <Form.Item label="Tokens per minute Limit (TPM)" name="tpm_limit">
                <InputNumber step={1} width={400} />
              </Form.Item>
              <Form.Item
                label="Requests per minute Limit (RPM)"
                name="rpm_limit"
              >
                <InputNumber step={1} width={400} />
              </Form.Item>
              <Form.Item label="Duration (eg: 30s, 30h, 30d)" name="duration">
                <Input />
              </Form.Item>
              <Form.Item label="Metadata" name="metadata">
                <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
              </Form.Item>
            </>
          ) : (
            <>
              <Form.Item label="Key Name" name="key_alias">
                <Input />
              </Form.Item>
              <Form.Item label="Team ID (Contact Group)" name="team_id">
                <Input placeholder="ai_team" />
              </Form.Item>

              <Form.Item label="Description" name="description">
                <Input.TextArea placeholder="Enter description" rows={4} />
              </Form.Item>
            </>
          )}
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Create Key</Button2>
          </div>
        </Form>
      </Modal>
      {apiKey && (
        <Modal
          title="Save your key"
          visible={isModalVisible}
          onOk={handleOk}
          onCancel={handleCancel}
          footer={null}
        >
          <Grid numItems={1} className="gap-2 w-full">
            <Col numColSpan={1}>
              <p>
                Please save this secret key somewhere safe and accessible. For
                security reasons, <b>you will not be able to view it again</b>{" "}
                through your LiteLLM account. If you lose this secret key, you
                will need to generate a new one.
              </p>
            </Col>
            <Col numColSpan={1}>
              {apiKey != null ? (
                <Text>API Key: {apiKey}</Text>
              ) : (
                <Text>Key being created, this might take 30s</Text>
              )}
            </Col>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

export default CreateKey;
