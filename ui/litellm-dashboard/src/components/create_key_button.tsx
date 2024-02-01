
"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { Card, Metric, Text } from "@tremor/react";
import { Button as Button2, Modal, Form, Input, InputNumber, Select, message } from "antd";
import { keyCreateCall } from "./networking";

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  accessToken: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  accessToken,
  data,
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
      // Check if "models" exists and is not an empty string
      if (formValues.models && formValues.models.trim() !== '') {
        // Format the "models" field as an array
        formValues.models = formValues.models.split(',').map((model: string) => model.trim());
      } else {
        // If "models" is undefined or an empty string, set it to an empty array
        formValues.models = [];
      }
      setIsModalVisible(true);
      const response = await keyCreateCall(accessToken, userID, formValues);
      setData((prevData) => (prevData ? [...prevData, response] : [response])); // Check if prevData is null
      setApiKey(response["key"]);
      message.success("API Key Created");
      form.resetFields();
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
        <Form form={form} onFinish={handleCreate} labelCol={{ span: 6 }} wrapperCol={{ span: 16 }} labelAlign="left">
        
          <Form.Item
            label="Key Name"
            name="key_alias"
          >
            <Input />
          </Form.Item>

          <Form.Item
            label="Models (Comma Separated). Eg: gpt-3.5-turbo,gpt-4"
            name="models"
          >
            <Input placeholder="gpt-4,gpt-3.5-turbo" />
          </Form.Item>
          

          <Form.Item
            label="Max Budget (USD)"
            name="max_budget"
          >
            <InputNumber step={0.01} precision={2} width={200}/>
          </Form.Item>
          <Form.Item
            label="Duration (eg: 30s, 30h, 30d)"
            name="duration"
          >
            <Input />
          </Form.Item>
          <Form.Item
            label="Team ID"
            name="team_id"
          >
            <Input placeholder="ai_team" />
          </Form.Item>
          <Form.Item
            label="Metadata"
            name="metadata"
          >
            <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
          </Form.Item>
          <div style={{ textAlign: 'right', marginTop: '10px' }}>
            <Button2 htmlType="submit">
              Create Key
            </Button2>
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
              through your LiteLLM account. If you lose this secret key, you will
              need to generate a new one.
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
