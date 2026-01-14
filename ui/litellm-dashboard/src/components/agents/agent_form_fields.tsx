import React from "react";
import { Form, Input, Switch, Collapse } from "antd";
import { Button as AntButton } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import { AGENT_FORM_CONFIG, SKILL_FIELD_CONFIG } from "./agent_config";

import CostConfigFields from "./cost_config_fields";

const { Panel } = Collapse;

interface AgentFormFieldsProps {
  showAgentName?: boolean;
}

/**
 * Reusable form fields component for agent forms
 * Uses shared configuration from agent_config.ts
 */
const AgentFormFields: React.FC<AgentFormFieldsProps> = ({ showAgentName = true }) => {
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

        {/* Skills */}
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
                <AntButton 
                  type="dashed" 
                  onClick={() => add()} 
                  icon={<PlusOutlined />}
                  style={{ width: '100%' }}
                >
                  Add Skill
                </AntButton>
              </>
            )}
          </Form.List>
        </Panel>

        {/* Capabilities */}
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

        {/* Optional Settings */}
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

        {/* Cost Configuration */}
        <Panel header={AGENT_FORM_CONFIG.cost.title} key={AGENT_FORM_CONFIG.cost.key}>
          <CostConfigFields />
        </Panel>

        {/* LiteLLM Parameters */}
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
      </Collapse>
    </>
  );
};

export default AgentFormFields;

