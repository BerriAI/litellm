// PROTOTYPE: 3-column env-vars editor (name, value, scope) for the create-MCP
// modal. Mounted as a plain Form.List under field name "mock_env_vars" so the
// caller can persist the values to localStorage on submit.

import React from "react";
import { Form, Input, Select, Button, Tooltip, Tag, Typography } from "antd";
import {
  InfoCircleOutlined,
  MinusCircleOutlined,
  PlusOutlined,
} from "@ant-design/icons";

const { Text } = Typography;

interface EnvVarsSectionProps {
  // When true, the value column is hidden — used by the Template modal where
  // variables only declare name + scope. Instance creation (default) keeps
  // the value column visible.
  templateMode?: boolean;
}

const EnvVarsSection: React.FC<EnvVarsSectionProps> = ({ templateMode = false }) => {
  return (
    <div className="rounded-lg border border-dashed border-purple-300 bg-purple-50 p-4">
      <div className="flex items-center gap-2 mb-1">
        <Tag color="purple" style={{ marginRight: 0 }}>
          Prototype
        </Tag>
        <Text strong className="text-sm">
          Variables
        </Text>
        <Tooltip
          title={
            <>
              Define variables you can interpolate in Static Headers or
              Authentication using <code>{"${VAR_NAME}"}</code>. <br />
              <b>Instance</b>: admin-defined value used for every user.
              <br />
              <b>Per-user</b>: each user supplies their own value (e.g. personal
              credentials).
            </>
          }
        >
          <InfoCircleOutlined className="text-purple-500" />
        </Tooltip>
      </div>
      <Text className="text-xs text-gray-600 block mb-3">
        {templateMode ? (
          <>
            A template declares <i>which</i> variables exist, not what
            they&apos;re set to. <b>Instance</b> values are entered when
            someone creates an instance from this template; <b>per-user</b>{" "}
            values come from each user&apos;s Variables tab at runtime.
            Reference them with <code>{"${VAR_NAME}"}</code> in Static
            Headers, URL, or Authentication.
          </>
        ) : (
          <>
            Reference these in Static Headers or Authentication as{" "}
            <code>{"${VAR_NAME}"}</code>. For example:{" "}
            <code className="bg-white px-1 rounded border border-gray-200">
              {"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}"}
            </code>
          </>
        )}
      </Text>

      <Form.List name="mock_env_vars">
        {(fields, { add, remove }) => (
          <div className="space-y-2">
            {fields.length > 0 && (
              <div className="flex gap-3 px-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
                <div style={{ flex: 1 }}>Variable Name</div>
                {!templateMode && <div style={{ flex: 1 }}>Value</div>}
                <div style={{ width: 160 }}>Scope</div>
                <div style={{ width: 24 }} />
              </div>
            )}
            {fields.map(({ key, name, ...restField }) => (
              <div key={key} className="flex gap-3 items-baseline">
                <Form.Item
                  {...restField}
                  name={[name, "name"]}
                  className="mb-0"
                  style={{ flex: 1 }}
                  rules={[
                    {
                      pattern: /^[A-Z_][A-Z0-9_]*$/,
                      message: "Use UPPER_SNAKE_CASE",
                      warningOnly: true,
                    },
                  ]}
                >
                  <Input
                    placeholder="e.g. DB_PROTOCOL"
                    className="rounded-md font-mono"
                  />
                </Form.Item>
                {!templateMode && (
                  <Form.Item
                    {...restField}
                    name={[name, "value"]}
                    className="mb-0"
                    style={{ flex: 1 }}
                    shouldUpdate
                  >
                    <ValueField fieldName={name} />
                  </Form.Item>
                )}
                <Form.Item
                  {...restField}
                  name={[name, "scope"]}
                  className="mb-0"
                  initialValue="instance"
                  style={{ width: 160 }}
                >
                  <Select
                    options={[
                      { value: "instance", label: "Instance" },
                      { value: "per_user", label: "Per-user" },
                    ]}
                  />
                </Form.Item>
                <div
                  style={{ width: 24 }}
                  className="flex items-center justify-center"
                >
                  <MinusCircleOutlined
                    onClick={() => remove(name)}
                    className="text-gray-500 hover:text-red-500 cursor-pointer"
                  />
                </div>
              </div>
            ))}
            <Button
              type="dashed"
              onClick={() => add({ scope: "instance" })}
              icon={<PlusOutlined />}
              block
            >
              Add Variable
            </Button>
          </div>
        )}
      </Form.List>
    </div>
  );
};

// Disables the value field when scope=per_user (those values come from each
// user later), keeping the column visible so the row layout stays consistent.
const ValueField: React.FC<{
  fieldName: number;
  value?: string;
  onChange?: (v: string) => void;
}> = ({ fieldName, value, onChange }) => {
  const scope = Form.useWatch(["mock_env_vars", fieldName, "scope"]);
  const isPerUser = scope === "per_user";
  return (
    <Input
      value={value ?? ""}
      onChange={(e) => onChange?.(e.target.value)}
      placeholder={
        isPerUser ? "Defined per user" : "e.g. postgresql"
      }
      disabled={isPerUser}
      className="rounded-md font-mono"
    />
  );
};

export default EnvVarsSection;
