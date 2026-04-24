import React, { useState } from "react";
import { Form, Input, Switch, Collapse, Select, Space, Tooltip, Modal, List, Tag, message } from "antd";
import { Button as AntButton } from "antd";
import { PlusOutlined, MinusCircleOutlined, InfoCircleOutlined, GithubOutlined, AppstoreOutlined, CheckCircleOutlined, CloseCircleOutlined, LoadingOutlined } from "@ant-design/icons";
import { AGENT_FORM_CONFIG, SKILL_FIELD_CONFIG } from "./agent_config";
import { fetchSkillsRegistry, testGitHubSkillConnection, createSkillFromGitHub, SkillRegistryItem } from "../networking";

import CostConfigFields from "./cost_config_fields";

const { Panel } = Collapse;

interface AgentFormFieldsProps {
  showAgentName?: boolean;
  visiblePanels?: string[];
  accessToken?: string | null;
}

const AgentFormFields: React.FC<AgentFormFieldsProps> = ({ showAgentName = true, visiblePanels, accessToken }) => {
  const shouldShow = (key: string) => !visiblePanels || visiblePanels.includes(key);

  // Registry modal state
  const [registryOpen, setRegistryOpen] = useState(false);
  const [registryItems, setRegistryItems] = useState<SkillRegistryItem[]>([]);
  const [registryLoading, setRegistryLoading] = useState(false);

  // GitHub import modal state
  const [githubOpen, setGithubOpen] = useState(false);
  const [githubRepoUrl, setGithubRepoUrl] = useState("");
  const [githubPat, setGithubPat] = useState("");
  const [testStatus, setTestStatus] = useState<"idle" | "loading" | "ok" | "error">("idle");
  const [testMessage, setTestMessage] = useState("");
  const [importLoading, setImportLoading] = useState(false);

  const openRegistry = async () => {
    if (!accessToken) return;
    setRegistryOpen(true);
    setRegistryLoading(true);
    const items = await fetchSkillsRegistry(accessToken);
    setRegistryItems(items);
    setRegistryLoading(false);
  };

  const handleTestConnection = async () => {
    if (!accessToken || !githubRepoUrl || !githubPat) return;
    setTestStatus("loading");
    const result = await testGitHubSkillConnection(accessToken, githubRepoUrl, githubPat);
    if (result.status === "ok") {
      setTestStatus("ok");
      setTestMessage("");
    } else {
      setTestStatus("error");
      setTestMessage(result.message ?? "Connection failed");
    }
  };

  const handleGitHubImport = async () => {
    if (!accessToken) return;
    setImportLoading(true);
    try {
      await createSkillFromGitHub(accessToken, githubRepoUrl, githubPat);
      message.success("Skill imported from GitHub successfully");
      setGithubOpen(false);
      setGithubRepoUrl("");
      setGithubPat("");
      setTestStatus("idle");
    } catch (e: any) {
      message.error(`Import failed: ${e.message}`);
    } finally {
      setImportLoading(false);
    }
  };

  const resetGithubModal = () => {
    setGithubOpen(false);
    setGithubRepoUrl("");
    setGithubPat("");
    setTestStatus("idle");
    setTestMessage("");
  };

  return (
    <>
      {showAgentName && (
        <Form.Item
          label="Agent Name"
          name="agent_name"
          rules={[{ required: true, message: "Please enter a unique agent name" }]}
          tooltip="Unique identifier for the agent"
        >
          <Input placeholder="e.g., customer-support-agent" />
        </Form.Item>
      )}

      <Collapse defaultActiveKey={['basic']} style={{ marginBottom: 16 }}>
        {/* Basic Information */}
        {shouldShow(AGENT_FORM_CONFIG.basic.key) && (
        <Panel header={`${AGENT_FORM_CONFIG.basic.title} (Required)`} key={AGENT_FORM_CONFIG.basic.key}>
          {AGENT_FORM_CONFIG.basic.fields.map((field) => (
            <Form.Item
              key={field.name}
              label={field.label}
              name={field.name}
              rules={field.required ? [{ required: true, message: `Please enter ${field.label.toLowerCase()}` }] : undefined}
              tooltip={field.tooltip}
            >
              {field.type === 'textarea' ? (
                <Input.TextArea rows={field.rows} placeholder={field.placeholder} />
              ) : (
                <Input placeholder={field.placeholder} />
              )}
            </Form.Item>
          ))}
        </Panel>
        )}

        {/* Skills */}
        {shouldShow(AGENT_FORM_CONFIG.skills.key) && (
        <Panel header={`${AGENT_FORM_CONFIG.skills.title} (Required)`} key={AGENT_FORM_CONFIG.skills.key}>
          <Form.List name="skills">
            {(fields, { add, remove }) => (
              <>
                {fields.map((field) => (
                  <div key={field.key} style={{ marginBottom: 16, padding: 16, border: '1px solid #d9d9d9', borderRadius: 4 }}>
                    <Form.Item
                      {...field}
                      label={SKILL_FIELD_CONFIG.id.label}
                      name={[field.name, 'id']}
                      rules={[{ required: SKILL_FIELD_CONFIG.id.required, message: 'Required' }]}
                    >
                      <Input placeholder={SKILL_FIELD_CONFIG.id.placeholder} />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label={SKILL_FIELD_CONFIG.name.label}
                      name={[field.name, 'name']}
                      rules={[{ required: SKILL_FIELD_CONFIG.name.required, message: 'Required' }]}
                    >
                      <Input placeholder={SKILL_FIELD_CONFIG.name.placeholder} />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label={SKILL_FIELD_CONFIG.description.label}
                      name={[field.name, 'description']}
                      rules={[{ required: SKILL_FIELD_CONFIG.description.required, message: 'Required' }]}
                    >
                      <Input.TextArea rows={SKILL_FIELD_CONFIG.description.rows} placeholder={SKILL_FIELD_CONFIG.description.placeholder} />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label={SKILL_FIELD_CONFIG.tags.label}
                      name={[field.name, 'tags']}
                      rules={[{ required: SKILL_FIELD_CONFIG.tags.required, message: 'Required' }]}
                      getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim())}
                      getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : value })}
                    >
                      <Input placeholder={SKILL_FIELD_CONFIG.tags.placeholder} />
                    </Form.Item>

                    <Form.Item
                      {...field}
                      label={SKILL_FIELD_CONFIG.examples.label}
                      name={[field.name, 'examples']}
                      getValueFromEvent={(e) => e.target.value.split(',').map((s: string) => s.trim()).filter((s: string) => s)}
                      getValueProps={(value) => ({ value: Array.isArray(value) ? value.join(', ') : '' })}
                    >
                      <Input placeholder={SKILL_FIELD_CONFIG.examples.placeholder} />
                    </Form.Item>

                    <AntButton
                      type="link"
                      danger
                      onClick={() => remove(field.name)}
                      icon={<MinusCircleOutlined />}
                    >
                      Remove Skill
                    </AntButton>
                  </div>
                ))}

                <Space style={{ width: '100%' }} direction="vertical">
                  <AntButton
                    type="dashed"
                    onClick={() => add()}
                    icon={<PlusOutlined />}
                    style={{ width: '100%' }}
                  >
                    Add Skill Manually
                  </AntButton>
                  {accessToken && (
                    <Space style={{ width: '100%' }}>
                      <AntButton
                        icon={<AppstoreOutlined />}
                        style={{ flex: 1 }}
                        onClick={openRegistry}
                      >
                        Browse Registry
                      </AntButton>
                      <AntButton
                        icon={<GithubOutlined />}
                        style={{ flex: 1 }}
                        onClick={() => setGithubOpen(true)}
                      >
                        Import from GitHub
                      </AntButton>
                    </Space>
                  )}
                </Space>

                {/* Registry browser modal */}
                <Modal
                  title="Skill Registry"
                  open={registryOpen}
                  onCancel={() => setRegistryOpen(false)}
                  footer={null}
                  width={600}
                >
                  <List
                    loading={registryLoading}
                    dataSource={registryItems}
                    locale={{ emptyText: "No skills in registry" }}
                    renderItem={(item) => (
                      <List.Item
                        actions={[
                          <AntButton
                            type="primary"
                            size="small"
                            key="add"
                            onClick={() => {
                              add({
                                id: item.skill_id,
                                name: item.display_title ?? item.skill_id,
                                description: item.description ?? "",
                                tags: item.tags,
                                examples: item.examples,
                              });
                              setRegistryOpen(false);
                              message.success(`Added "${item.display_title ?? item.skill_id}"`);
                            }}
                          >
                            Add
                          </AntButton>,
                        ]}
                      >
                        <List.Item.Meta
                          title={item.display_title ?? item.skill_id}
                          description={
                            <>
                              <div style={{ marginBottom: 4 }}>{item.description}</div>
                              {item.tags.map((t) => <Tag key={t}>{t}</Tag>)}
                            </>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </Modal>

                {/* GitHub import modal */}
                <Modal
                  title="Import Skill from GitHub"
                  open={githubOpen}
                  onCancel={resetGithubModal}
                  footer={[
                    <AntButton key="cancel" onClick={resetGithubModal}>
                      Cancel
                    </AntButton>,
                    <AntButton
                      key="import"
                      type="primary"
                      disabled={testStatus !== "ok"}
                      loading={importLoading}
                      onClick={handleGitHubImport}
                    >
                      Import Skill
                    </AntButton>,
                  ]}
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Input
                      placeholder="https://github.com/org/my-skill"
                      value={githubRepoUrl}
                      onChange={(e) => { setGithubRepoUrl(e.target.value); setTestStatus("idle"); }}
                      addonBefore="Repo URL"
                    />
                    <Input.Password
                      placeholder="ghp_xxxxxxxxxxxx"
                      value={githubPat}
                      onChange={(e) => { setGithubPat(e.target.value); setTestStatus("idle"); }}
                      addonBefore="GitHub PAT"
                    />
                    <AntButton
                      onClick={handleTestConnection}
                      loading={testStatus === "loading"}
                      disabled={!githubRepoUrl || !githubPat}
                    >
                      Test Connection
                    </AntButton>
                    {testStatus === "ok" && (
                      <span style={{ color: "#52c41a" }}>
                        <CheckCircleOutlined /> Connected successfully
                      </span>
                    )}
                    {testStatus === "error" && (
                      <span style={{ color: "#ff4d4f" }}>
                        <CloseCircleOutlined /> {testMessage}
                      </span>
                    )}
                  </Space>
                </Modal>
              </>
            )}
          </Form.List>
        </Panel>
        )}

        {/* Capabilities */}
        {shouldShow(AGENT_FORM_CONFIG.capabilities.key) && (
        <Panel header={AGENT_FORM_CONFIG.capabilities.title} key={AGENT_FORM_CONFIG.capabilities.key}>
          {AGENT_FORM_CONFIG.capabilities.fields.map((field) => (
            <Form.Item
              key={field.name}
              label={field.label}
              name={field.name}
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          ))}
        </Panel>
        )}

        {/* Optional Settings */}
        {shouldShow(AGENT_FORM_CONFIG.optional.key) && (
        <Panel header={AGENT_FORM_CONFIG.optional.title} key={AGENT_FORM_CONFIG.optional.key}>
          {AGENT_FORM_CONFIG.optional.fields.map((field) => (
            <Form.Item
              key={field.name}
              label={field.label}
              name={field.name}
              valuePropName={field.type === 'switch' ? 'checked' : undefined}
            >
              {field.type === 'switch' ? <Switch /> : <Input placeholder={field.placeholder} />}
            </Form.Item>
          ))}
        </Panel>
        )}

        {/* Cost Configuration */}
        {shouldShow(AGENT_FORM_CONFIG.cost.key) && (
        <Panel header={AGENT_FORM_CONFIG.cost.title} key={AGENT_FORM_CONFIG.cost.key}>
          <CostConfigFields />
        </Panel>
        )}

        {/* LiteLLM Parameters */}
        {shouldShow(AGENT_FORM_CONFIG.litellm.key) && (
        <Panel header={AGENT_FORM_CONFIG.litellm.title} key={AGENT_FORM_CONFIG.litellm.key}>
          {AGENT_FORM_CONFIG.litellm.fields.map((field) => (
            <Form.Item
              key={field.name}
              label={field.label}
              name={field.name}
              valuePropName={field.type === 'switch' ? 'checked' : undefined}
            >
              {field.type === 'switch' ? <Switch /> : <Input placeholder={field.placeholder} />}
            </Form.Item>
          ))}
        </Panel>
        )}

        {/* Authentication Headers */}
        {shouldShow("auth_headers") && (
        <Panel header="Authentication Headers" key="auth_headers">
          {/* Static Headers */}
          <Form.Item
            label={
              <span>
                Static Headers{" "}
                <Tooltip title="Headers always sent to the backend agent, regardless of the client request. Admin-configured, static wins on conflict.">
                  <InfoCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
          >
            <Form.List name="static_headers">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <Space key={key} style={{ display: "flex", marginBottom: 8 }} align="baseline">
                      <Form.Item
                        {...restField}
                        name={[name, "header"]}
                        rules={[{ required: true, message: "Header name required" }]}
                      >
                        <Input placeholder="Header name (e.g. Authorization)" style={{ width: 220 }} />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, "value"]}
                        rules={[{ required: true, message: "Value required" }]}
                      >
                        <Input placeholder="Value (e.g. Bearer token123)" style={{ width: 260 }} />
                      </Form.Item>
                      <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ff4d4f" }} />
                    </Space>
                  ))}
                  <AntButton type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{ width: "100%" }}>
                    Add Static Header
                  </AntButton>
                </>
              )}
            </Form.List>
          </Form.Item>

          {/* Extra Headers (dynamic forwarding) */}
          <Form.Item
            label={
              <span>
                Forward Client Headers{" "}
                <Tooltip title="Header names to extract from the client's request and forward to the agent. Type a name and press Enter.">
                  <InfoCircleOutlined style={{ color: "#8c8c8c" }} />
                </Tooltip>
              </span>
            }
            name="extra_headers"
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder="e.g. x-api-key, Authorization"
              tokenSeparators={[","]}
            />
          </Form.Item>
        </Panel>
        )}
      </Collapse>
    </>
  );
};

export default AgentFormFields;
