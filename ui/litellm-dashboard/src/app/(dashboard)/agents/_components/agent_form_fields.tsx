import React from "react";
import { Form, Input, Switch, Collapse, Select, Space, Tooltip } from "antd";
import { Button as AntButton } from "antd";
import { PlusOutlined, MinusCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { AGENT_FORM_CONFIG, SKILL_FIELD_CONFIG } from "./agent_config";

import CostConfigFields from "./cost_config_fields";
import { useTranslation } from "react-i18next";

const { Panel } = Collapse;

interface AgentFormFieldsProps {
  showAgentName?: boolean;
  visiblePanels?: string[];
}

/**
 * Reusable form fields component for agent forms
 * Uses shared configuration from agent_config.ts
 */
const AgentFormFields: React.FC<AgentFormFieldsProps> = ({ showAgentName = true, visiblePanels }) => {
  const { t } = useTranslation("gateway");
  const shouldShow = (key: string) => !visiblePanels || visiblePanels.includes(key);
  const sections: Record<string, string> = {
    basic: t("agents.form.sections.basic"),
    skills: t("agents.form.sections.skills"),
    capabilities: t("agents.form.sections.capabilities"),
    optional: t("agents.form.sections.optional"),
    litellm: t("agents.form.sections.litellm"),
    cost: t("agents.form.sections.cost"),
  };
  const fieldCopy: Record<string, { label: string; placeholder?: string; tooltip?: string; helpText?: string }> = {
    name: {
      label: t("agents.form.fields.name.label"),
      placeholder: t("agents.form.fields.name.placeholder"),
    },
    description: {
      label: t("agents.form.fields.description.label"),
      placeholder: t("agents.form.fields.description.placeholder"),
    },
    url: { label: t("agents.form.fields.url.label"), tooltip: t("agents.form.fields.url.tooltip") },
    version: { label: t("agents.form.fields.version.label") },
    protocolVersion: {
      label: t("agents.form.fields.protocolVersion.label"),
      tooltip: t("agents.form.fields.protocolVersion.tooltip"),
      helpText: t("agents.form.fields.protocolVersion.help"),
    },
    streaming: { label: t("agents.form.fields.streaming.label") },
    pushNotifications: { label: t("agents.form.fields.pushNotifications.label") },
    stateTransitionHistory: { label: t("agents.form.fields.stateTransitionHistory.label") },
    iconUrl: { label: t("agents.form.fields.iconUrl.label") },
    documentationUrl: { label: t("agents.form.fields.documentationUrl.label") },
    supportsAuthenticatedExtendedCard: {
      label: t("agents.form.fields.supportsAuthenticatedExtendedCard.label"),
    },
    model: { label: t("agents.form.fields.model.label") },
    make_public: { label: t("agents.form.fields.make_public.label") },
  };
  return (
    <>
      {showAgentName && (
        <Form.Item
          label={t("agents.form.agentName")}
          name="agent_name"
          rules={[{ required: true, message: t("agents.form.agentNameRequired") }]}
          tooltip={t("agents.form.agentNameHint")}
        >
          <Input placeholder={t("agents.form.agentNamePlaceholder")} />
        </Form.Item>
      )}

      <Collapse defaultActiveKey={["basic"]} style={{ marginBottom: 16 }}>
        {/* Basic Information */}
        {shouldShow(AGENT_FORM_CONFIG.basic.key) && (
          <Panel header={`${sections.basic} (${t("agents.form.requiredSuffix")})`} key={AGENT_FORM_CONFIG.basic.key}>
            {AGENT_FORM_CONFIG.basic.fields.map((field) => {
              const copy = fieldCopy[field.name] ?? field;
              return (
                <Form.Item
                  key={field.name}
                  label={copy.label}
                  name={field.name}
                  rules={
                    field.required
                      ? [{ required: true, message: t("agents.form.requiredField", { field: copy.label }) }]
                      : undefined
                  }
                  tooltip={copy.tooltip}
                  extra={copy.helpText}
                >
                  {field.type === "textarea" ? (
                    <Input.TextArea rows={field.rows} placeholder={copy.placeholder ?? field.placeholder} />
                  ) : field.type === "select" ? (
                    <Select placeholder={copy.placeholder ?? field.placeholder}>
                      {(field.options ?? []).map((opt) => (
                        <Select.Option key={opt} value={opt}>
                          {opt}
                        </Select.Option>
                      ))}
                    </Select>
                  ) : (
                    <Input placeholder={copy.placeholder ?? field.placeholder} />
                  )}
                </Form.Item>
              );
            })}
          </Panel>
        )}

        {/* Skills */}
        {shouldShow(AGENT_FORM_CONFIG.skills.key) && (
          <Panel header={sections.skills} key={AGENT_FORM_CONFIG.skills.key}>
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
                        label={t("agents.form.skill.id")}
                        name={[field.name, "id"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.id.required, message: t("agents.form.required") }]}
                      >
                        <Input placeholder={t("agents.form.skill.idPlaceholder")} />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={t("agents.form.skill.name")}
                        name={[field.name, "name"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.name.required, message: t("agents.form.required") }]}
                      >
                        <Input placeholder={t("agents.form.skill.namePlaceholder")} />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={t("agents.form.skill.description")}
                        name={[field.name, "description"]}
                        rules={[
                          { required: SKILL_FIELD_CONFIG.description.required, message: t("agents.form.required") },
                        ]}
                      >
                        <Input.TextArea
                          rows={SKILL_FIELD_CONFIG.description.rows}
                          placeholder={t("agents.form.skill.descriptionPlaceholder")}
                        />
                      </Form.Item>

                      <Form.Item
                        {...field}
                        label={t("agents.form.skill.tags")}
                        name={[field.name, "tags"]}
                        rules={[{ required: SKILL_FIELD_CONFIG.tags.required, message: t("agents.form.required") }]}
                      >
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          tokenSeparators={[","]}
                          placeholder={t("agents.form.skill.tagsPlaceholder")}
                        />
                      </Form.Item>

                      <Form.Item {...field} label={t("agents.form.skill.examples")} name={[field.name, "examples"]}>
                        <Select
                          mode="tags"
                          style={{ width: "100%" }}
                          tokenSeparators={[","]}
                          placeholder={t("agents.form.skill.examplesPlaceholder")}
                        />
                      </Form.Item>

                      <AntButton type="link" danger onClick={() => remove(field.name)} icon={<MinusCircleOutlined />}>
                        {t("agents.form.skill.remove")}
                      </AntButton>
                    </div>
                  ))}
                  <AntButton type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{ width: "100%" }}>
                    {t("agents.form.skill.add")}
                  </AntButton>
                </>
              )}
            </Form.List>
          </Panel>
        )}

        {/* Capabilities */}
        {shouldShow(AGENT_FORM_CONFIG.capabilities.key) && (
          <Panel header={sections.capabilities} key={AGENT_FORM_CONFIG.capabilities.key}>
            {AGENT_FORM_CONFIG.capabilities.fields.map((field) => (
              <Form.Item
                key={field.name}
                label={fieldCopy[field.name]?.label ?? field.label}
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
          <Panel header={sections.optional} key={AGENT_FORM_CONFIG.optional.key}>
            {AGENT_FORM_CONFIG.optional.fields.map((field) => (
              <Form.Item
                key={field.name}
                label={fieldCopy[field.name]?.label ?? field.label}
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
          <Panel header={sections.cost} key={AGENT_FORM_CONFIG.cost.key}>
            <CostConfigFields />
          </Panel>
        )}

        {/* LiteLLM Parameters */}
        {shouldShow(AGENT_FORM_CONFIG.litellm.key) && (
          <Panel header={sections.litellm} key={AGENT_FORM_CONFIG.litellm.key}>
            {AGENT_FORM_CONFIG.litellm.fields.map((field) => (
              <Form.Item
                key={field.name}
                label={fieldCopy[field.name]?.label ?? field.label}
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
          <Panel header={t("agents.form.authHeaders.title")} key="auth_headers">
            {/* Static Headers */}
            <Form.Item
              label={
                <span>
                  {t("agents.form.authHeaders.static")}{" "}
                  <Tooltip title={t("agents.form.authHeaders.staticHint")}>
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
                          rules={[{ required: true, message: t("agents.form.authHeaders.nameRequired") }]}
                        >
                          <Input placeholder={t("agents.form.authHeaders.namePlaceholder")} style={{ width: 220 }} />
                        </Form.Item>
                        <Form.Item
                          {...restField}
                          name={[name, "value"]}
                          rules={[{ required: true, message: t("agents.form.authHeaders.valueRequired") }]}
                        >
                          <Input placeholder={t("agents.form.authHeaders.valuePlaceholder")} style={{ width: 260 }} />
                        </Form.Item>
                        <MinusCircleOutlined onClick={() => remove(name)} style={{ color: "#ff4d4f" }} />
                      </Space>
                    ))}
                    <AntButton type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{ width: "100%" }}>
                      {t("agents.form.authHeaders.add")}
                    </AntButton>
                  </>
                )}
              </Form.List>
            </Form.Item>

            {/* Extra Headers (dynamic forwarding) */}
            <Form.Item
              label={
                <span>
                  {t("agents.form.authHeaders.forward")}{" "}
                  <Tooltip title={t("agents.form.authHeaders.forwardHint")}>
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
