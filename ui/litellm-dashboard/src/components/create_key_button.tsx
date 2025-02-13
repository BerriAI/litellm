"use client";

import React, { useState, useEffect, useRef } from "react";
import { Button, TextInput, Grid, Col } from "@tremor/react";
import {
  Card,
  Metric,
  Text,
  Title,
  Subtitle,
  Accordion,
  AccordionHeader,
  AccordionBody,
} from "@tremor/react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  message,
  Radio,
} from "antd";
import { unfurlWildcardModelsInList, getModelDisplayName } from "./key_team_helpers/fetch_available_models_team_key";
import {
  keyCreateCall,
  slackBudgetAlertsHealthCheck,
  modelAvailableCall,
  getGuardrailsList,
} from "./networking";
import { InfoCircleOutlined } from '@ant-design/icons';
import { Tooltip } from 'antd';

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  team: any | null;
  userRole: string | null;
  accessToken: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
}

const getPredefinedTags = (data: any[] | null) => {
  let allTags = [];

  console.log("data:", JSON.stringify(data));

  if (data) {
    for (let key of data) {
      if (key["metadata"] && key["metadata"]["tags"]) {
        allTags.push(...key["metadata"]["tags"]);
      }
    }
  }

  // Deduplicate using Set
  const uniqueTags = Array.from(new Set(allTags)).map(tag => ({
    value: tag,
    label: tag,
  }));


  console.log("uniqueTags:", uniqueTags);
  return uniqueTags;
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
  const [userModels, setUserModels] = useState<string[]>([]);
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);
  const [keyOwner, setKeyOwner] = useState("you");
  const [predefinedTags, setPredefinedTags] = useState(getPredefinedTags(data));
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);

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
          const model_available = await modelAvailableCall(
            accessToken,
            userID,
            userRole
          );
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

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name
        );
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  const handleCreate = async (formValues: Record<string, any>) => {
    try {
      const newKeyAlias = formValues?.key_alias ?? "";
      const newKeyTeamId = formValues?.team_id ?? null;

      const existingKeyAliases =
        data
          ?.filter((k) => k.team_id === newKeyTeamId)
          .map((k) => k.key_alias) ?? [];

      if (existingKeyAliases.includes(newKeyAlias)) {
        throw new Error(
          `Key alias ${newKeyAlias} already exists for team with ID ${newKeyTeamId}, please provide another key alias`
        );
      }

      message.info("Making API Call");
      setIsModalVisible(true);
      
      if(keyOwner === "you"){
        formValues.user_id = userID 
      }
      // If it's a service account, add the service_account_id to the metadata
      if (keyOwner === "service_account") {
        // Parse existing metadata or create an empty object
        let metadata: Record<string, any> = {};
        try {
          metadata = JSON.parse(formValues.metadata || "{}");
        } catch (error) {
          console.error("Error parsing metadata:", error);
        }
        metadata["service_account_id"] = formValues.key_alias;
        // Update the formValues with the new metadata
        formValues.metadata = JSON.stringify(metadata);
      }

      const response = await keyCreateCall(accessToken, userID, formValues);

      console.log("key create Response:", response);
      setData((prevData) => (prevData ? [...prevData, response] : [response])); // Check if prevData is null
      setApiKey(response["key"]);
      setSoftBudget(response["soft_budget"]);
      message.success("API Key Created");
      form.resetFields();
      localStorage.removeItem("userData" + userID);
    } catch (error) {
      console.log("error in create key:", error);
      message.error(`Error creating the key: ${error}`);
    }
  };

  const handleCopy = () => {
    message.success("API Key copied to clipboard");
  };

  useEffect(() => {
    let tempModelsToPick = [];

    if (team) {
      if (team.models.length > 0) {
        if (team.models.includes("all-proxy-models")) {
          // if the team has all-proxy-models show all available models
          tempModelsToPick = userModels;
        } else {
          // show team models
          tempModelsToPick = team.models;
        }
      } else {
        // show all available models if the team has no models set
        tempModelsToPick = userModels;
      }
    } else {
      // no team set, show all available models
      tempModelsToPick = userModels;
    }

    tempModelsToPick = unfurlWildcardModelsInList(tempModelsToPick, userModels);

    setModelsToPick(tempModelsToPick);
  }, [team, userModels]);

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
          <>
            <Form.Item label="Owned By" className="mb-4">
              <Radio.Group
                onChange={(e) => setKeyOwner(e.target.value)}
                value={keyOwner}
              >
                <Radio value="you">You</Radio>
                <Radio value="service_account">Service Account</Radio>
                {userRole === "Admin" && <Radio value="another_user">Another User</Radio>}
              </Radio.Group>
            </Form.Item>

            <Form.Item
              label="User ID"
              name="user_id"
              hidden={keyOwner !== "another_user"}
              valuePropName="user_id"
              className="mt-8"
              rules={[{ required: keyOwner === "another_user", message: `Please input the user ID of the user you are assigning the key to` }]}
              help={"Get User ID - Click on the 'Users' tab in the sidebar."}
            >
              <TextInput 
                placeholder="User ID" 
                onChange={(e) => form.setFieldValue('user_id', e.target.value)}
              />
            </Form.Item>

            <Form.Item
              label={keyOwner === "you" || keyOwner === "another_user" ? "Key Name" : "Service Account ID"}
              name="key_alias"
              rules={[{ required: true, message: `Please input a ${keyOwner === "you" ? "key name" : "service account ID"}` }]}
              help={keyOwner === "you" ? "required" : "IDs can include letters, numbers, and hyphens"}
            >
              <TextInput placeholder="" />
            </Form.Item>
            <Form.Item
              label="Team ID"
              name="team_id"
              hidden={keyOwner !== "another_user"}
              initialValue={team ? team["team_id"] : null}
              valuePropName="team_id"
              className="mt-8"
            >
              <TextInput defaultValue={team ? team["team_id"] : null} onChange={(e) => form.setFieldValue('team_id', e.target.value)}/>
            </Form.Item>

            <Form.Item
              label="Models"
              name="models"
              rules={[{ required: true, message: "Please select a model" }]}
              help="required"
            >
              <Select
                mode="multiple"
                placeholder="Select models"
                style={{ width: "100%" }}
                onChange={(values) => {
                  // Check if "All Team Models" is selected
                  const isAllTeamModelsSelected =
                    values.includes("all-team-models");

                  // If "All Team Models" is selected, deselect all other models
                  if (isAllTeamModelsSelected) {
                    const newValues = ["all-team-models"];
                    // You can call the form's setFieldsValue method to update the value
                    form.setFieldsValue({ models: newValues });
                  }
                }}
              >
                <Option key="all-team-models" value="all-team-models">
                  All Team Models
                </Option>
                {modelsToPick.map((model: string) => (
                  <Option key={model} value={model}>
                    {getModelDisplayName(model)}
                  </Option>
                ))}
              </Select>
            </Form.Item>
            <Accordion className="mt-20 mb-8">
              <AccordionHeader>
                <b>Optional Settings</b>
              </AccordionHeader>
              <AccordionBody>
                <Form.Item
                  className="mt-8"
                  label="Max Budget (USD)"
                  name="max_budget"
                  help={`Budget cannot exceed team max budget: $${team?.max_budget !== null && team?.max_budget !== undefined ? team?.max_budget : "unlimited"}`}
                  rules={[
                    {
                      validator: async (_, value) => {
                        if (
                          value &&
                          team &&
                          team.max_budget !== null &&
                          value > team.max_budget
                        ) {
                          throw new Error(
                            `Budget cannot exceed team max budget: $${team.max_budget}`
                          );
                        }
                      },
                    },
                  ]}
                >
                  <InputNumber step={0.01} precision={2} width={200} />
                </Form.Item>
                <Form.Item
                  className="mt-8"
                  label="Reset Budget"
                  name="budget_duration"
                  help={`Team Reset Budget: ${team?.budget_duration !== null && team?.budget_duration !== undefined ? team?.budget_duration : "None"}`}
                >
                  <Select defaultValue={null} placeholder="n/a">
                    <Select.Option value="24h">daily</Select.Option>
                    <Select.Option value="7d">weekly</Select.Option>
                    <Select.Option value="30d">monthly</Select.Option>
                  </Select>
                </Form.Item>
                <Form.Item
                  className="mt-8"
                  label="Tokens per minute Limit (TPM)"
                  name="tpm_limit"
                  help={`TPM cannot exceed team TPM limit: ${team?.tpm_limit !== null && team?.tpm_limit !== undefined ? team?.tpm_limit : "unlimited"}`}
                  rules={[
                    {
                      validator: async (_, value) => {
                        if (
                          value &&
                          team &&
                          team.tpm_limit !== null &&
                          value > team.tpm_limit
                        ) {
                          throw new Error(
                            `TPM limit cannot exceed team TPM limit: ${team.tpm_limit}`
                          );
                        }
                      },
                    },
                  ]}
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
                <Form.Item
                  className="mt-8"
                  label="Requests per minute Limit (RPM)"
                  name="rpm_limit"
                  help={`RPM cannot exceed team RPM limit: ${team?.rpm_limit !== null && team?.rpm_limit !== undefined ? team?.rpm_limit : "unlimited"}`}
                  rules={[
                    {
                      validator: async (_, value) => {
                        if (
                          value &&
                          team &&
                          team.rpm_limit !== null &&
                          value > team.rpm_limit
                        ) {
                          throw new Error(
                            `RPM limit cannot exceed team RPM limit: ${team.rpm_limit}`
                          );
                        }
                      },
                    },
                  ]}
                >
                  <InputNumber step={1} width={400} />
                </Form.Item>
                <Form.Item
                  label="Expire Key (eg: 30s, 30h, 30d)"
                  name="duration"
                  className="mt-8"
                >
                  <TextInput placeholder="" />
                </Form.Item>
                <Form.Item 
                  label={
                    <span>
                      Guardrails{' '}
                      <Tooltip title="Setup your first guardrail">
                        <a 
                          href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start" 
                          target="_blank" 
                          rel="noopener noreferrer"
                          onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                        >
                          <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                        </a>
                      </Tooltip>
                    </span>
                  }
                  name="guardrails" 
                  className="mt-8"
                  help="Select existing guardrails or enter new ones"
                >
                  <Select
                    mode="tags"
                    style={{ width: '100%' }}
                    placeholder="Select or enter guardrails"
                    options={guardrailsList.map(name => ({ value: name, label: name }))}
                  />
                </Form.Item>

                <Form.Item label="Metadata" name="metadata" className="mt-8">
                  <Input.TextArea
                    rows={4}
                    placeholder="Enter metadata as JSON"
                  />
                </Form.Item>
                <Form.Item label="Tags" name="tags" className="mt-8" help={`Tags for tracking spend and/or doing tag-based routing.`}>
                <Select
                    mode="tags"
                    style={{ width: '100%' }}
                    placeholder="Enter tags"
                    tokenSeparators={[',']}
                    options={predefinedTags}
                  />
                </Form.Item>
              </AccordionBody>
            </Accordion>
          </>

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
                  <Text className="mt-3">API Key:</Text>
                  <div
                    style={{
                      background: "#f8f8f8",
                      padding: "10px",
                      borderRadius: "5px",
                      marginBottom: "10px",
                    }}
                  >
                    <pre
                      style={{ wordWrap: "break-word", whiteSpace: "normal" }}
                    >
                      {apiKey}
                    </pre>
                  </div>

                  <CopyToClipboard text={apiKey} onCopy={handleCopy}>
                    <Button className="mt-3">Copy API Key</Button>
                  </CopyToClipboard>
                  {/* <Button className="mt-3" onClick={sendSlackAlert}>
                    Test Key
                </Button> */}
                </div>
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
