"use client";
import React, { useState, useEffect, useRef, useCallback } from "react";
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
import SchemaFormFields from './common_components/check_openapi_schema';
import {
  keyCreateCall,
  slackBudgetAlertsHealthCheck,
  modelAvailableCall,
  getGuardrailsList,
  proxyBaseUrl,
  getPossibleUserRoles,
  userFilterUICall,
} from "./networking";
import { Team } from "./key_team_helpers/key_list";
import TeamDropdown from "./common_components/team_dropdown";
import { InfoCircleOutlined } from '@ant-design/icons';
import { Tooltip } from 'antd';
import Createuser from "./create_user_button";
import debounce from 'lodash/debounce';

const { Option } = Select;

interface CreateKeyProps {
  userID: string;
  team: Team | null;
  userRole: string | null;
  accessToken: string;
  data: any[] | null;
  setData: React.Dispatch<React.SetStateAction<any[] | null>>;
  teams: Team[] | null;
}

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
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

export const getTeamModels = (team: Team | null, allAvailableModels: string[]): string[] => {
  let tempModelsToPick = [];

  if (team) {
    if (team.models.length > 0) {
      if (team.models.includes("all-proxy-models")) {
        // if the team has all-proxy-models show all available models
        tempModelsToPick = allAvailableModels;
      } else {
        // show team models
        tempModelsToPick = team.models;
      }
    } else {
      // show all available models if the team has no models set
      tempModelsToPick = allAvailableModels;
    }
  } else {
    // no team set, show all available models
    tempModelsToPick = allAvailableModels;
  }

  return unfurlWildcardModelsInList(tempModelsToPick, allAvailableModels);
};

export const fetchUserModels = async (userID: string, userRole: string, accessToken: string, setUserModels: (models: string[]) => void) => {
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

const CreateKey: React.FC<CreateKeyProps> = ({
  userID,
  team,
  teams,
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
  const [selectedCreateKeyTeam, setSelectedCreateKeyTeam] = useState<Team | null>(team);
  const [isCreateUserModalVisible, setIsCreateUserModalVisible] = useState(false);
  const [newlyCreatedUserId, setNewlyCreatedUserId] = useState<string | null>(null);
  const [possibleUIRoles, setPossibleUIRoles] = useState<
    Record<string, Record<string, string>>
  >({});
  const [userOptions, setUserOptions] = useState<UserOption[]>([]);
  const [userSearchLoading, setUserSearchLoading] = useState<boolean>(false);

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
    if (userID && userRole && accessToken) {
      fetchUserModels(userID, userRole, accessToken, setUserModels);
    }
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

  // Fetch possible user roles when component mounts
  useEffect(() => {
    const fetchPossibleRoles = async () => {
      try {
        if (accessToken) {
          // Check if roles are cached in session storage
          const cachedRoles = sessionStorage.getItem('possibleUserRoles');
          if (cachedRoles) {
            setPossibleUIRoles(JSON.parse(cachedRoles));
          } else {
            const availableUserRoles = await getPossibleUserRoles(accessToken);
            sessionStorage.setItem('possibleUserRoles', JSON.stringify(availableUserRoles));
            setPossibleUIRoles(availableUserRoles);
          }
        }
      } catch (error) {
        console.error("Error fetching possible user roles:", error);
      }
    };
    
    fetchPossibleRoles();
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
    const models = getTeamModels(selectedCreateKeyTeam, userModels);
    setModelsToPick(models);
    form.setFieldValue('models', []);
  }, [selectedCreateKeyTeam, userModels]);

  // Add a callback function to handle user creation
  const handleUserCreated = (userId: string) => {
    setNewlyCreatedUserId(userId);
    form.setFieldsValue({ user_id: userId });
    setIsCreateUserModalVisible(false);
  };

  const fetchUsers = async (searchText: string): Promise<void> => {
    if (!searchText) {
      setUserOptions([]);
      return;
    }

    setUserSearchLoading(true);
    try {
      const params = new URLSearchParams();
      params.append('user_email', searchText); // Always search by email
      if (accessToken == null) {
        return;
      }
      const response = await userFilterUICall(accessToken, params);
      
      const data: User[] = response;
      const options: UserOption[] = data.map(user => ({
        label: `${user.user_email} (${user.user_id})`,
        value: user.user_id,
        user
      }));
      
      setUserOptions(options);
    } catch (error) {
      console.error('Error fetching users:', error);
      message.error('Failed to search for users');
    } finally {
      setUserSearchLoading(false);
    }
  };

  const debouncedSearch = useCallback(
    debounce((text: string) => fetchUsers(text), 300),
    [accessToken]
  );

  const handleUserSearch = (value: string): void => {
    debouncedSearch(value);
  };

  const handleUserSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_id: selectedUser.user_id
    });
  };

  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
        + Create New Key
      </Button>
      <Modal
        // title="Create Key"
        visible={isModalVisible}
        width={1000}
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
          {/* Section 1: Key Ownership */}
          <div className="mb-8">
            <Title className="mb-4">Key Ownership</Title>
            <Form.Item 
              label={
                <span>
                  Owned By{' '}
                  <Tooltip title="Select who will own this API key">
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              } 
              className="mb-4"
            >
              <Radio.Group
                onChange={(e) => setKeyOwner(e.target.value)}
                value={keyOwner}
              >
                <Radio value="you">You</Radio>
                <Radio value="service_account">Service Account</Radio>
                {userRole === "Admin" && <Radio value="another_user">Another User</Radio>}
              </Radio.Group>
            </Form.Item>

            {keyOwner === "another_user" && (
              <Form.Item
                label={
                  <span>
                    User ID{' '}
                    <Tooltip title="The user who will own this key and be responsible for its usage">
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                    </Tooltip>
                  </span>
                }
                name="user_id"
                className="mt-4"
                rules={[{ required: keyOwner === "another_user", message: `Please input the user ID of the user you are assigning the key to` }]}
              >
                <div>
                  <div style={{ display: 'flex', marginBottom: '8px' }}>
                    <Select
                      showSearch
                      placeholder="Type email to search for users"
                      filterOption={false}
                      onSearch={handleUserSearch}
                      onSelect={(value, option) => handleUserSelect(value, option as UserOption)}
                      options={userOptions}
                      loading={userSearchLoading}
                      allowClear
                      style={{ width: '100%' }}
                      notFoundContent={userSearchLoading ? 'Searching...' : 'No users found'}
                    />
                    <Button2 
                      onClick={() => setIsCreateUserModalVisible(true)}
                      style={{ marginLeft: '8px' }}
                    >
                      Create User
                    </Button2>
                  </div>
                  <div className="text-xs text-gray-500">
                    Search by email to find users
                  </div>
                </div>
              </Form.Item>
            )}
            <Form.Item
              label={
                <span>
                  Team{' '}
                  <Tooltip title="The team this key belongs to, which determines available models and budget limits">
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              }
              name="team_id"
              initialValue={team ? team.team_id : null}
              className="mt-4"
            >
              <TeamDropdown 
                teams={teams} 
                onChange={(teamId) => {
                  const selectedTeam = teams?.find(t => t.team_id === teamId) || null;
                  setSelectedCreateKeyTeam(selectedTeam);
                }}
              />
            </Form.Item>

          </div>

          {/* Section 2: Key Details */}
          <div className="mb-8">
            <Title className="mb-4">Key Details</Title>
            <Form.Item
              label={
                <span>
                  {keyOwner === "you" || keyOwner === "another_user" ? "Key Name" : "Service Account ID"}{' '}
                  <Tooltip title={keyOwner === "you" || keyOwner === "another_user" ? 
                    "A descriptive name to identify this key" : 
                    "Unique identifier for this service account"}>
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              }
              name="key_alias"
              rules={[{ required: true, message: `Please input a ${keyOwner === "you" ? "key name" : "service account ID"}` }]}
              help="required"
            >
              <TextInput placeholder="" />
            </Form.Item>
            
            <Form.Item
              label={
                <span>
                  Models{' '}
                  <Tooltip title="Select which models this key can access. Choose 'All Team Models' to grant access to all models available to the team">
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              }
              name="models"
              rules={[{ required: true, message: "Please select a model" }]}
              help="required"
              className="mt-4"
            >
              <Select
                mode="multiple"
                placeholder="Select models"
                style={{ width: "100%" }}
                onChange={(values) => {
                  if (values.includes("all-team-models")) {
                    form.setFieldsValue({ models: ["all-team-models"] });
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
          </div>

          {/* Section 3: Optional Settings */}
          <div className="mb-8">
            <Accordion className="mt-4 mb-4">
              <AccordionHeader>
                <Title className="m-0">Optional Settings</Title>
              </AccordionHeader>
              <AccordionBody>
                <Form.Item
                  className="mt-4"
                  label={
                    <span>
                      Max Budget (USD){' '}
                      <Tooltip title="Maximum amount in USD this key can spend. When reached, the key will be blocked from making further requests">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  }
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
                  className="mt-4"
                  label={
                    <span>
                      Reset Budget{' '}
                      <Tooltip title="How often the budget should reset. For example, setting 'daily' will reset the budget every 24 hours">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  }
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
                  className="mt-4"
                  label={
                    <span>
                      Tokens per minute Limit (TPM){' '}
                      <Tooltip title="Maximum number of tokens this key can process per minute. Helps control usage and costs">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  }
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
                  className="mt-4"
                  label={
                    <span>
                      Requests per minute Limit (RPM){' '}
                      <Tooltip title="Maximum number of API requests this key can make per minute. Helps prevent abuse and manage load">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  }
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
                  label={
                    <span>
                      Expire Key{' '}
                      <Tooltip title="Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days)">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  }
                  name="duration"
                  className="mt-4"
                >
                  <TextInput placeholder="e.g., 30d" />
                </Form.Item>
                <Form.Item 
                  label={
                    <span>
                      Guardrails{' '}
                      <Tooltip title="Apply safety guardrails to this key to filter content or enforce policies">
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
                  className="mt-4"
                  help="Select existing guardrails or enter new ones"
                >
                  <Select
                    mode="tags"
                    style={{ width: '100%' }}
                    placeholder="Select or enter guardrails"
                    options={guardrailsList.map(name => ({ value: name, label: name }))}
                  />
                </Form.Item>

                <Form.Item 
                  label={
                    <span>
                      Metadata{' '}
                      <Tooltip title="JSON object with additional information about this key. Used for tracking or custom logic">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  } 
                  name="metadata" 
                  className="mt-4"
                >
                  <Input.TextArea
                    rows={4}
                    placeholder="Enter metadata as JSON"
                  />
                </Form.Item>
                <Form.Item 
                  label={
                    <span>
                      Tags{' '}
                      <Tooltip title="Tags for tracking spend and/or doing tag-based routing. Used for analytics and filtering">
                        <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                      </Tooltip>
                    </span>
                  } 
                  name="tags" 
                  className="mt-4" 
                  help={`Tags for tracking spend and/or doing tag-based routing.`}
                >
                <Select
                    mode="tags"
                    style={{ width: '100%' }}
                    placeholder="Enter tags"
                    tokenSeparators={[',']}
                    options={predefinedTags}
                  />
                </Form.Item>
                <Accordion className="mt-4 mb-4">
                  <AccordionHeader>
                  <div className="flex items-center gap-2">

                    <b>Advanced Settings</b>
                    <Tooltip title={ 
                      <span>
                        Learn more about advanced settings in our{' '}
                        <a 
                          href={proxyBaseUrl ? `${proxyBaseUrl}/#/key%20management/generate_key_fn_key_generate_post`: `/#/key%20management/generate_key_fn_key_generate_post`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-400 hover:text-blue-300"
                        >
                          documentation
                        </a>
                      </span>
                    }>
                      <InfoCircleOutlined className="text-gray-400 hover:text-gray-300 cursor-help" />
                    </Tooltip>
                    </div>
                  </AccordionHeader>
                  <AccordionBody>
                    <SchemaFormFields 
                      schemaComponent="GenerateKeyRequest"
                      form={form}
                      excludedFields={['key_alias', 'team_id', 'models', 'duration', 'metadata', 'tags', 'guardrails', "max_budget", "budget_duration", "tpm_limit", "rpm_limit"]}
                    />
                  </AccordionBody>
                </Accordion>
              </AccordionBody>
            </Accordion>
          </div>

          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Create Key</Button2>
          </div>
        </Form>
      </Modal>

      {/* Add the Create User Modal */}
      {isCreateUserModalVisible && (
        <Modal
          title="Create New User"
          visible={isCreateUserModalVisible}
          onCancel={() => setIsCreateUserModalVisible(false)}
          footer={null}
          width={800}
        >
          <Createuser 
            userID={userID}
            accessToken={accessToken}
            teams={teams}
            possibleUIRoles={possibleUIRoles}
            onUserCreated={handleUserCreated}
            isEmbedded={true}
          />
        </Modal>
      )}

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
