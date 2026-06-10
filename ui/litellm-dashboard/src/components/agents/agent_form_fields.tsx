import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Form, Input, Switch, Collapse, Select, Space, Tooltip } from "antd";
import { Button as AntButton } from "antd";
import { PlusOutlined, MinusCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { getAgentFormConfig, getSkillFieldConfig } from "./agent_config";

import CostConfigFields from "./cost_config_fields";

const { Panel } = Collapse;

interface AgentFormFieldsProps {
  showAgentName?: boolean;
  visiblePanels?: string[];
}

const AgentFormFields: React.FC<AgentFormFieldsProps> = ({ showAgentName = true, visiblePanels }) => {
  const { t } = useTranslation();
  const AGENT_FORM_CONFIG = useMemo(() => getAgentFormConfig(t), [t]);
  const SKILL_FIELD_CONFIG = useMemo(() => getSkillFieldConfig(t), [t]);

  const shouldShow = (key: string) => !visiblePanels || visiblePanels.includes(key);
  return (
    <>
      {showAgentName && (
        <Form.Item
          label={t("agentsPage.agentFormFields.agentNameLabel")}
          name="agent_name"
          rules={[{ required: true, message: t("agentsPage.agentFormFields.agentNameRequired") }]}
          tooltip={t("agentsPage.agentFormFields.agentNameTooltip")}
        >
          <Input placeholder={t("agentsPage.agentFormFields.agentNamePlaceholder")} />
        </Form.Item>
      )}

      <Collapse defaultActiveKey={["basic"]} style={{ marginBottom: 16 }}>
        {/* Basic Information */}
        {shouldShow(AGENT_FORM_CONFIG.basic.key) && (
          <Panel
            header={`${AGENT_FORM_CONFIG.basic.title} (${t("common.required")})`}
            key={AGENT_FORM_CONFIG.basic.key}
          >
            {AGENT_FORM_CONFIG.basic.fields.map((field) => (
              <Form.Item
                key={field.name}
                label={field.label}
                name={field.name}
                rules={
                  field.required
                    ? [
                        {
                          required: true,
                          message: t("agentsPage.agentFormFields.pleaseEnterField", {
                            field: field.label.toLowerCase(),
                          }),
                        },
                      ]
                    : undefined
                }
                tooltip={field.tooltip}
              >
                {field.type === "textarea" ? (
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
          <Panel header={AGENT_FORM_CONFIG.skills.title} key={AGENT_FORM_CONFIG.skills.key}>
            <Form.List name="skills">
              {(fields, { add, remove }) => (
                <>
                  {fields.map((field) => (
                    <div
                      key={field.key}
                      style={{ marginBottom: 16, padding: 16, border: "1px solid #d9d9d9", borderRadius: 4 }}
                    >
                      <Form.Item
                        {...field}
                        label={SKILL_FIELD_CONFIG.id.label}
                        name={[field.name, "id"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.id.required, message: t("common.required") }]}
                      >
                        <Input placeholder={SKILL_FIELD_CONFIG.id.placeholder} />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={SKILL_FIELD_CONFIG.name.label}
                        name={[field.name, "name"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.name.required, message: t("common.required") }]}
                      >
                        <Input placeholder={SKILL_FIELD_CONFIG.name.placeholder} />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={SKILL_FIELD_CONFIG.description.label}
                        name={[field.name, "description"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.description.required, message: t("common.required") }]}
                      >
                        <Input.TextArea
                          rows={SKILL_FIELD_CONFIG.description.rows}
                          placeholder={SKILL_FIELD_CONFIG.description.placeholder}
                        />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={SKILL_FIELD_CONFIG.tags.label}
                        name={[field.name, "tags"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.tags.required, message: t("common.required") }]}
                      >
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          tokenSeparators={[","]}
                          placeholder={SKILL_FIELD_CONFIG.tags.placeholder}
                        />
                      </Form.Item>

                      <Form.Item {...field} label={SKILL_FIELD_CONFIG.examples.label} name={[field.name, "examples"]}>
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          tokenSeparators={[","]}
                          placeholder={SKILL_FIELD_CONFIG.examples.placeholder}
                        />
                      </Form.Item>

                      <AntButton type="link" danger onClick={() => remove(field.name)} icon={<MinusCircleOutlined />}>
                        {t("agentsPage.agentFormFields.removeSkill")}
                      </AntButton>
                    </div>
                  ))}
                  <AntButton type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{ width: "100%" }}>
                    {t("agentsPage.agentFormFields.addSkill")}
                  </AntButton>
                </>
              )}
            </Form.List>
          </Panel>
        )}

        {/* Capabilities */}
        {shouldShow(AGENT_FORM_CONFIG.capabilities.key) && (
          <Panel header={AGENT_FORM_CONFIG.capabilities.title} key={AGENT_FORM_CONFIG.capabilities.key}>
            {AGENT_FORM_CONFIG.capabilities.fields.map((field) => (
              <Form.Item key={field.name} label={field.label} name={field.name} valuePropName="checked">
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
                valuePropName={field.type === "switch" ? "checked" : undefined}
              >
                {field.type === "switch" ? <Switch /> : <Input placeholder={field.placeholder} />}
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
                valuePropName={field.type === "switch" ? "checked" : undefined}
              >
                {field.type === "switch" ? <Switch /> : <Input placeholder={field.placeholder} />}
              </Form.Item>
            ))}
          </Panel>
        )}

        {/* Authentication Headers */}
        {shouldShow("auth_headers") && (
          <Panel header={t("agentsPage.agentFormFields.authHeadersTitle")} key="auth_headers">
            {/* Static Headers */}
            <Form.Item
              label={
                <span>
                  {t("agentsPage.agentFormFields.staticHeadersLabel")}{" "}
                  <Tooltip title={t("agentsPage.agentFormFields.staticHeadersTooltip")}>
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
                          rules={[{ required: true, message: t("agentsPage.agentFormFields.headerNameRequired") }]}
                        >
                          <Input
                            placeholder={t("agentsPage.agentFormFields.headerNamePlaceholder")}
                            style={{ width: 220 }}
                          />
                        </Form.Item>
                        <Form.Item
                          {...restField}
                          name={[name, "value"]}
                          rules={[{ required: true, message: t("agentsPage.agentFormFields.headerValueRequired") }]}
                        >
                          <Input
                            placeholder={t("agentsPage.agentFormFields.headerValuePlaceholder")}
                            style={{ width: 260 }}
                          />
                        </Form.Item>
                        <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ff4d4f" }} />
                      </Space>
                    ))}
                    <AntButton type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{ width: "100%" }}>
                      {t("agentsPage.agentFormFields.addStaticHeader")}
                    </AntButton>
                  </>
                )}
              </Form.List>
            </Form.Item>

            {/* Extra Headers (dynamic forwarding) */}
            <Form.Item
              label={
                <span>
                  {t("agentsPage.agentFormFields.forwardClientHeadersLabel")}{" "}
                  <Tooltip title={t("agentsPage.agentFormFields.forwardClientHeadersTooltip")}>
                    <InfoCircleOutlined style={{ color: "#8c8c8c" }} />
                  </Tooltip>
                </span>
              }
              name="extra_headers"
            >
              <Select
                mode="tags"
                style={{ width: "100%" }}
                placeholder={t("agentsPage.agentFormFields.forwardClientHeadersPlaceholder")}
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
