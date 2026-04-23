import React from "react";
import { Form, Switch, Collapse, Select } from "antd";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info, MinusCircle, Plus } from "lucide-react";
import { AGENT_FORM_CONFIG, SKILL_FIELD_CONFIG } from "./agent_config";

import CostConfigFields from "./cost_config_fields";

const { Panel } = Collapse;

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 inline h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

interface AgentFormFieldsProps {
  showAgentName?: boolean;
  visiblePanels?: string[];
}

/**
 * Reusable form fields component for agent forms
 * Uses shared configuration from agent_config.ts
 */
const AgentFormFields: React.FC<AgentFormFieldsProps> = ({ showAgentName = true, visiblePanels }) => {
  const shouldShow = (key: string) => !visiblePanels || visiblePanels.includes(key);
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
                <Textarea rows={field.rows} placeholder={field.placeholder} />
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
                      <Textarea rows={SKILL_FIELD_CONFIG.description.rows} placeholder={SKILL_FIELD_CONFIG.description.placeholder} />
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
                    
                    <Button
                      type="button"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      onClick={() => remove(field.name)}
                    >
                      <MinusCircle className="h-4 w-4" />
                      Remove Skill
                    </Button>
                  </div>
                ))}
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => add()}
                  className="w-full border-dashed"
                >
                  <Plus className="h-4 w-4" />
                  Add Skill
                </Button>
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
                <InfoTip>
                  Headers always sent to the backend agent, regardless of the
                  client request. Admin-configured, static wins on conflict.
                </InfoTip>
              </span>
            }
          >
            <Form.List name="static_headers">
              {(fields, { add, remove }) => (
                <>
                  {fields.map(({ key, name, ...restField }) => (
                    <div
                      key={key}
                      className="flex items-baseline gap-2 mb-2"
                    >
                      <Form.Item
                        {...restField}
                        name={[name, "header"]}
                        rules={[
                          {
                            required: true,
                            message: "Header name required",
                          },
                        ]}
                        className="mb-0"
                      >
                        <Input
                          placeholder="Header name (e.g. Authorization)"
                          className="w-[220px]"
                        />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, "value"]}
                        rules={[
                          { required: true, message: "Value required" },
                        ]}
                        className="mb-0"
                      >
                        <Input
                          placeholder="Value (e.g. Bearer token123)"
                          className="w-[260px]"
                        />
                      </Form.Item>
                      <button
                        type="button"
                        onClick={() => remove(name)}
                        className="text-destructive"
                        aria-label="Remove header"
                      >
                        <MinusCircle className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => add()}
                    className="w-full border-dashed"
                  >
                    <Plus className="h-4 w-4" />
                    Add Static Header
                  </Button>
                </>
              )}
            </Form.List>
          </Form.Item>

          {/* Extra Headers (dynamic forwarding) */}
          <Form.Item
            label={
              <span>
                Forward Client Headers{" "}
                <InfoTip>
                  Header names to extract from the client&apos;s request and
                  forward to the agent. Type a name and press Enter.
                </InfoTip>
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

