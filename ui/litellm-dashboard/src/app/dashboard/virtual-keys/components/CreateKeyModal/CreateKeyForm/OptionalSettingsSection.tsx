"use client";

import { Accordion, AccordionBody, AccordionHeader, Text, Title } from "@tremor/react";
import { Form, FormInstance, Input, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import NumericalInput from "@/components/shared/numerical_input";
import BudgetDurationDropdown from "@/components/common_components/budget_duration_dropdown";
import RateLimitTypeFormItem from "@/components/common_components/RateLimitTypeFormItem";
import VectorStoreSelector from "@/components/vector_store_management/VectorStoreSelector";
import MCPServerSelector from "@/components/mcp_server_management/MCPServerSelector";
import PremiumLoggingSettings from "@/components/common_components/PremiumLoggingSettings";
import ModelAliasManager from "@/components/common_components/ModelAliasManager";
import KeyLifecycleSettings from "@/components/common_components/KeyLifecycleSettings";
import { proxyBaseUrl } from "@/components/networking";
import SchemaFormFields from "@/components/common_components/check_openapi_schema";
import { Team } from "@/components/key_team_helpers/key_list";
import React from "react";
import { DefaultOptionType } from "rc-select/lib/Select";
import { ModelAliases } from "@/app/dashboard/virtual-keys/components/CreateKeyModal/types";

export interface OptionalSettingsSectionProps {
  form: FormInstance;
  team: Team | null;
  premiumUser: boolean;
  guardrails: string[];
  prompts: string[];
  accessToken: string;
  predefinedTags: DefaultOptionType[];
  loggingSettings: any;
  setLoggingSettings: (settings: any) => void;
  disabledCallbacks: string[];
  setDisabledCallbacks: (disabledCallbacks: string[]) => void;
  modelAliases: ModelAliases;
  setModelAliases: (aliases: ModelAliases) => void;
  autoRotationEnabled: boolean;
  setAutoRotationEnabled: (enabled: boolean) => void;
  rotationInterval: string;
  setRotationInterval: (interval: string) => void;
}

// TODO: break apart this component into smaller sections
const OptionalSettingsSection = ({
  form,
  team,
  premiumUser,
  guardrails,
  prompts,
  accessToken,
  predefinedTags,
  loggingSettings,
  setLoggingSettings,
  disabledCallbacks,
  setDisabledCallbacks,
  modelAliases,
  setModelAliases,
  autoRotationEnabled,
  setAutoRotationEnabled,
  rotationInterval,
  setRotationInterval,
}: OptionalSettingsSectionProps) => {
  return (
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
                Max Budget (USD){" "}
                <Tooltip title="Maximum amount in USD this key can spend. When reached, the key will be blocked from making further requests">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="max_budget"
            help={`Budget cannot exceed team max budget: $${team?.max_budget !== null && team?.max_budget !== undefined ? team?.max_budget : "unlimited"}`}
            rules={[
              {
                validator: async (_, value) => {
                  if (value && team && team.max_budget !== null && value > team.max_budget) {
                    throw new Error(
                      `Budget cannot exceed team max budget: $${formatNumberWithCommas(team.max_budget, 4)}`,
                    );
                  }
                },
              },
            ]}
          >
            <NumericalInput step={0.01} precision={2} width={200} />
          </Form.Item>
          <Form.Item
            className="mt-4"
            label={
              <span>
                Reset Budget{" "}
                <Tooltip title="How often the budget should reset. For example, setting 'daily' will reset the budget every 24 hours">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="budget_duration"
            help={`Team Reset Budget: ${team?.budget_duration !== null && team?.budget_duration !== undefined ? team?.budget_duration : "None"}`}
          >
            <BudgetDurationDropdown onChange={(value) => form.setFieldValue("budget_duration", value)} />
          </Form.Item>
          <Form.Item
            className="mt-4"
            label={
              <span>
                Tokens per minute Limit (TPM){" "}
                <Tooltip title="Maximum number of tokens this key can process per minute. Helps control usage and costs">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="tpm_limit"
            help={`TPM cannot exceed team TPM limit: ${team?.tpm_limit !== null && team?.tpm_limit !== undefined ? team?.tpm_limit : "unlimited"}`}
            rules={[
              {
                validator: async (_, value) => {
                  if (value && team && team.tpm_limit !== null && value > team.tpm_limit) {
                    throw new Error(`TPM limit cannot exceed team TPM limit: ${team.tpm_limit}`);
                  }
                },
              },
            ]}
          >
            <NumericalInput step={1} width={400} />
          </Form.Item>
          <RateLimitTypeFormItem
            type="tpm"
            name="tpm_limit_type"
            className="mt-4"
            initialValue={null}
            form={form}
            showDetailedDescriptions={true}
          />
          <Form.Item
            className="mt-4"
            label={
              <span>
                Requests per minute Limit (RPM){" "}
                <Tooltip title="Maximum number of API requests this key can make per minute. Helps prevent abuse and manage load">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="rpm_limit"
            help={`RPM cannot exceed team RPM limit: ${team?.rpm_limit !== null && team?.rpm_limit !== undefined ? team?.rpm_limit : "unlimited"}`}
            rules={[
              {
                validator: async (_, value) => {
                  if (value && team && team.rpm_limit !== null && value > team.rpm_limit) {
                    throw new Error(`RPM limit cannot exceed team RPM limit: ${team.rpm_limit}`);
                  }
                },
              },
            ]}
          >
            <NumericalInput step={1} width={400} />
          </Form.Item>
          <RateLimitTypeFormItem
            type="rpm"
            name="rpm_limit_type"
            className="mt-4"
            initialValue={null}
            form={form}
            showDetailedDescriptions={true}
          />
          <Form.Item
            label={
              <span>
                Guardrails{" "}
                <Tooltip title="Apply safety guardrails to this key to filter content or enforce policies">
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                  >
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </a>
                </Tooltip>
              </span>
            }
            name="guardrails"
            className="mt-4"
            help={
              premiumUser
                ? "Select existing guardrails or enter new ones"
                : "Premium feature - Upgrade to set guardrails by key"
            }
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              disabled={!premiumUser}
              placeholder={
                !premiumUser ? "Premium feature - Upgrade to set guardrails by key" : "Select or enter guardrails"
              }
              options={guardrails.map((name) => ({ value: name, label: name }))}
            />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Prompts{" "}
                <Tooltip title="Allow this key to use specific prompt templates">
                  <a
                    href="https://docs.litellm.ai/docs/proxy/prompt_management"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()} // Prevent accordion from collapsing when clicking link
                  >
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </a>
                </Tooltip>
              </span>
            }
            name="prompts"
            className="mt-4"
            help={
              premiumUser
                ? "Select existing prompts or enter new ones"
                : "Premium feature - Upgrade to set prompts by key"
            }
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              disabled={!premiumUser}
              placeholder={!premiumUser ? "Premium feature - Upgrade to set prompts by key" : "Select or enter prompts"}
              options={prompts.map((name) => ({ value: name, label: name }))}
            />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Allowed Vector Stores{" "}
                <Tooltip title="Select which vector stores this key can access. If none selected, the key will have access to all available vector stores">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="allowed_vector_store_ids"
            className="mt-4"
            help="Select vector stores this key can access. Leave empty for access to all vector stores"
          >
            <VectorStoreSelector
              onChange={(values: string[]) => form.setFieldValue("allowed_vector_store_ids", values)}
              value={form.getFieldValue("allowed_vector_store_ids")}
              accessToken={accessToken}
              placeholder="Select vector stores (optional)"
            />
          </Form.Item>

          <Form.Item
            label={
              <span>
                Allowed MCP Servers{" "}
                <Tooltip title="Select which MCP servers or access groups this key can access. ">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="allowed_mcp_servers_and_groups"
            className="mt-4"
            help="Select MCP servers or access groups this key can access. "
          >
            <MCPServerSelector
              onChange={(val: any) => form.setFieldValue("allowed_mcp_servers_and_groups", val)}
              value={form.getFieldValue("allowed_mcp_servers_and_groups")}
              accessToken={accessToken}
              placeholder="Select MCP servers or access groups (optional)"
            />
          </Form.Item>

          <Form.Item
            label={
              <span>
                Metadata{" "}
                <Tooltip title="JSON object with additional information about this key. Used for tracking or custom logic">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="metadata"
            className="mt-4"
          >
            <Input.TextArea rows={4} placeholder="Enter metadata as JSON" />
          </Form.Item>
          <Form.Item
            label={
              <span>
                Tags{" "}
                <Tooltip title="Tags for tracking spend and/or doing tag-based routing. Used for analytics and filtering">
                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                </Tooltip>
              </span>
            }
            name="tags"
            className="mt-4"
            help={`Tags for tracking spend and/or doing tag-based routing.`}
          >
            <Select
              mode="tags"
              style={{ width: "100%" }}
              placeholder="Enter tags"
              tokenSeparators={[","]}
              options={predefinedTags}
            />
          </Form.Item>

          {premiumUser ? (
            <Accordion className="mt-4 mb-4">
              <AccordionHeader>
                <b>Logging Settings</b>
              </AccordionHeader>
              <AccordionBody>
                <div className="mt-4">
                  <PremiumLoggingSettings
                    value={loggingSettings}
                    onChange={setLoggingSettings}
                    premiumUser={true}
                    disabledCallbacks={disabledCallbacks}
                    onDisabledCallbacksChange={setDisabledCallbacks}
                  />
                </div>
              </AccordionBody>
            </Accordion>
          ) : (
            <Tooltip
              title={
                <span>
                  Key-level logging settings is an enterprise feature, get in touch -
                  <a href="https://www.litellm.ai/enterprise" target="_blank">
                    https://www.litellm.ai/enterprise
                  </a>
                </span>
              }
              placement="top"
            >
              <div style={{ position: "relative" }}>
                <div style={{ opacity: 0.5 }}>
                  <Accordion className="mt-4 mb-4">
                    <AccordionHeader>
                      <b>Logging Settings</b>
                    </AccordionHeader>
                    <AccordionBody>
                      <div className="mt-4">
                        <PremiumLoggingSettings
                          value={loggingSettings}
                          onChange={setLoggingSettings}
                          premiumUser={false}
                          disabledCallbacks={disabledCallbacks}
                          onDisabledCallbacksChange={setDisabledCallbacks}
                        />
                      </div>
                    </AccordionBody>
                  </Accordion>
                </div>
                <div style={{ position: "absolute", inset: 0, cursor: "not-allowed" }} />
              </div>
            </Tooltip>
          )}

          <Accordion className="mt-4 mb-4">
            <AccordionHeader>
              <b>Model Aliases</b>
            </AccordionHeader>
            <AccordionBody>
              <div className="mt-4">
                <Text className="text-sm text-gray-600 mb-4">
                  Create custom aliases for models that can be used in API calls. This allows you to create shortcuts
                  for specific models.
                </Text>
                <ModelAliasManager
                  accessToken={accessToken}
                  initialModelAliases={modelAliases}
                  onAliasUpdate={setModelAliases}
                  showExampleConfig={false}
                />
              </div>
            </AccordionBody>
          </Accordion>

          <Accordion className="mt-4 mb-4">
            <AccordionHeader>
              <b>Key Lifecycle</b>
            </AccordionHeader>
            <AccordionBody>
              <div className="mt-4">
                <KeyLifecycleSettings
                  form={form}
                  autoRotationEnabled={autoRotationEnabled}
                  onAutoRotationChange={setAutoRotationEnabled}
                  rotationInterval={rotationInterval}
                  onRotationIntervalChange={setRotationInterval}
                />
              </div>
            </AccordionBody>
          </Accordion>
          <Accordion className="mt-4 mb-4">
            <AccordionHeader>
              <div className="flex items-center gap-2">
                <b>Advanced Settings</b>
                <Tooltip
                  title={
                    <span>
                      Learn more about advanced settings in our{" "}
                      <a
                        href={
                          proxyBaseUrl
                            ? `${proxyBaseUrl}/#/key%20management/generate_key_fn_key_generate_post`
                            : `/#/key%20management/generate_key_fn_key_generate_post`
                        }
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:text-blue-300"
                      >
                        documentation
                      </a>
                    </span>
                  }
                >
                  <InfoCircleOutlined className="text-gray-400 hover:text-gray-300 cursor-help" />
                </Tooltip>
              </div>
            </AccordionHeader>
            <AccordionBody>
              <SchemaFormFields
                schemaComponent="GenerateKeyRequest"
                form={form}
                excludedFields={[
                  "key_alias",
                  "team_id",
                  "models",
                  "duration",
                  "metadata",
                  "tags",
                  "guardrails",
                  "max_budget",
                  "budget_duration",
                  "tpm_limit",
                  "rpm_limit",
                ]}
              />
            </AccordionBody>
          </Accordion>
        </AccordionBody>
      </Accordion>
    </div>
  );
};

export default OptionalSettingsSection;
