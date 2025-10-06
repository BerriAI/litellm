"use client";

import { TextInput, Title } from "@tremor/react";
import { Form, FormInstance, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import React from "react";

export interface KeyDetailsSectionProps {
  form: FormInstance;
  keyOwner: string;
  modelsToPick: string[];
  keyType: string;
  setKeyType: (keyType: string) => void;
}

const { Option } = Select;

const KeyDetailsSection = ({ form, keyOwner, keyType, modelsToPick, setKeyType }: KeyDetailsSectionProps) => {
  return (
    <div className="mb-8">
      <Title className="mb-4">Key Details</Title>
      <Form.Item
        label={
          <span>
            {keyOwner === "you" || keyOwner === "another_user" ? "Key Name" : "Service Account ID"}{" "}
            <Tooltip
              title={
                keyOwner === "you" || keyOwner === "another_user"
                  ? "A descriptive name to identify this key"
                  : "Unique identifier for this service account"
              }
            >
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="key_alias"
        rules={[
          {
            required: true,
            message: `Please input a ${keyOwner === "you" ? "key name" : "service account ID"}`,
          },
        ]}
        help="required"
      >
        <TextInput placeholder="" />
      </Form.Item>

      <Form.Item
        label={
          <span>
            Models{" "}
            <Tooltip title="Select which models this key can access. Choose 'All Team Models' to grant access to all models available to the team">
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="models"
        rules={
          keyType === "management" || keyType === "read_only"
            ? []
            : [{ required: true, message: "Please select a model" }]
        }
        help={
          keyType === "management" || keyType === "read_only"
            ? "Models field is disabled for this key type"
            : "required"
        }
        className="mt-4"
      >
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
          disabled={keyType === "management" || keyType === "read_only"}
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

      <Form.Item
        label={
          <span>
            Key Type{" "}
            <Tooltip title="Select the type of key to determine what routes and operations this key can access">
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="key_type"
        initialValue="default"
        className="mt-4"
      >
        <Select
          defaultValue="default"
          placeholder="Select key type"
          style={{ width: "100%" }}
          optionLabelProp="label"
          onChange={(value) => {
            setKeyType(value);
            // Clear models field and disable if management or read_only
            if (value === "management" || value === "read_only") {
              form.setFieldsValue({ models: [] });
            }
          }}
        >
          <Option value="default" label="Default">
            <div style={{ padding: "4px 0" }}>
              <div style={{ fontWeight: 500 }}>Default</div>
              <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                Can call LLM API + Management routes
              </div>
            </div>
          </Option>
          <Option value="llm_api" label="LLM API">
            <div style={{ padding: "4px 0" }}>
              <div style={{ fontWeight: 500 }}>LLM API</div>
              <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                Can call only LLM API routes (chat/completions, embeddings, etc.)
              </div>
            </div>
          </Option>
          <Option value="management" label="Management">
            <div style={{ padding: "4px 0" }}>
              <div style={{ fontWeight: 500 }}>Management</div>
              <div style={{ fontSize: "11px", color: "#6b7280", marginTop: "2px" }}>
                Can call only management routes (user/team/key management)
              </div>
            </div>
          </Option>
        </Select>
      </Form.Item>
    </div>
  );
};

export default KeyDetailsSection;
