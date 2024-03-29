"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import { Card, Metric, Text, Title, Subtitle } from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  message,
} from "antd";
import { keyCreateCall, slackBudgetAlertsHealthCheck, modelAvailableCall } from "./networking";

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  team: any | null;
  userRole: string | null;
  accessToken: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  team,
  userRole,
  accessToken,
  data,
  setData,
}) => {
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [apiKey, setApiKey] = useState(null);
  const [softBudget, setSoftBudget] = useState(null);
  const [userModels, setUserModels] = useState([]);
  const handleOk = () => {
    setIsModalVisible(false);
    form.resetFields();
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setApiKey(null);
    form.resetFields();
  };

  useEffect(() => {
    const fetchUserModels = async () => {
      try {
        if (userID === null || userRole === null) {
          return;
        }

        if (accessToken !== null) {
          const model_available = await modelAvailableCall(accessToken, userID, userRole);
          let available_model_names = model_available["data"].map(
            (element: { id: string }) => element.id
          );
          console.log("available_model_names:", available_model_names);
          setUserModels(available_model_names);
        }
      } catch (error) {
        console.error("Error fetching user models:", error);
      }
    };
  
    fetchUserModels();
  }, [accessToken, userID, userRole]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      message.info("Making API Call");
      setIsModalVisible(true);
      const response = await keyCreateCall(accessToken, userID, formValues);

      console.log("key create Response:", response);
      setData((prevData) => (prevData ? [...prevData, response] : [response])); // Check if prevData is null
      setApiKey(response["key"]);
      setSoftBudget(response["soft_budget"]);
      message.success("API Key Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.error("Error creating the key:", error);
    }
  };

  const sendSlackAlert = async () => {
    try {
      console.log("Sending Slack alert...");
      const response = await slackBudgetAlertsHealthCheck(accessToken);
      console.log("slackBudgetAlertsHealthCheck Response:", response);
      console.log("Testing Slack alert successful");
    } catch (error) {
      console.error("Error sending Slack alert:", error);
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
              <Form.Item 
                label="Key Name" 
                name="key_alias"
                rules={[{ required: true, message: 'Please input a key name' }]}
              >
                <Input />
              </Form.Item>
              <Form.Item
                label="Team ID"
                name="team_id"
                initialValue={team ? team["team_id"] : null}
                valuePropName="team_id"
              >
                <Input value={team ? team["team_alias"] : ""} disabled />
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
              <Form.Item label="Reset Budget" name="budget_duration">
                <Select defaultValue={null} placeholder="n/a">
                  <Select.Option value="24h">daily</Select.Option>
                  <Select.Option value="30d">monthly</Select.Option>
                </Select>
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
              <Form.Item label="Expire Key (eg: 30s, 30h, 30d)" name="duration">
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
                <Input placeholder="default team (create a new team)" />
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
          visible={isModalVisible}
          onOk={handleOk}
          onCancel={handleCancel}
          footer={null}
        >
          <Grid numItems={1} className="gap-2 w-full">
            <Card>
              <Title>Save your Key</Title>
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
                  <div>
                    <Text>API Key: {apiKey}</Text>
                    <Title className="mt-6">Budgets</Title>
                      <Text>Soft Limit Budget: ${softBudget}</Text>
                      <Button className="mt-3" onClick={sendSlackAlert}>
                        Test Slack Alert
                      </Button>
                      <Text className="mt-2">
                        (LiteLLM Docs - 
                        <a href="https://docs.litellm.ai/docs/proxy/alerting" target="_blank" className="text-blue-500">
                           Set Up Slack Alerting)
                        </a>
                      </Text>
                  </div>
                ) : (
                  <Text>Key being created, this might take 30s</Text>
                )}
              </Col>
            </Card>
          </Grid>
        </Modal>
      )}
    </div>
  );
};

export default CreateKey;
