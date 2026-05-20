// PROTOTYPE: 3-column env-vars editor (name, value, scope) for the create-MCP
// modal. Mounted as a plain Form.List under field name "mock_env_vars" so the
// caller can persist the values to localStorage on submit.

import React from "react";
import { Form, Input, Select, Space, Button, Tooltip, Tag, Typography } from "antd";
import {
  InfoCircleOutlined,
  MinusCircleOutlined,
  PlusOutlined,
} from "@ant-design/icons";

const { Text } = Typography;

const EnvVarsSection: React.FC = () => {
  return (
    <div className="rounded-lg border border-dashed border-purple-300 bg-purple-50 p-4">
      <div className="flex items-center gap-2 mb-1">
        <Tag color="purple" style={{ marginRight: 0 }}>
          Prototype
        </Tag>
        <Text strong className="text-sm">
          Environment Variables
        </Text>
        <Tooltip
          title={
            <>
              Define variables you can interpolate in Static Headers using{" "}
              <code>{"${VAR_NAME}"}</code>. <br />
              <b>Global</b>: admin-defined value used for every user.
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
        Reference these in Static Headers as <code>{"${VAR_NAME}"}</code>. For
        example:{" "}
        <code className="bg-white px-1 rounded border border-gray-200">
          {"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}"}
        </code>
      </Text>

      <Form.List name="mock_env_vars">
        {(fields, { add, remove }) => (
          <div className="space-y-2">
            {fields.length > 0 && (
              <div className="flex gap-3 px-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
                <div style={{ flex: 1 }}>Variable Name</div>
                <div style={{ flex: 1 }}>Value</div>
                <div style={{ width: 160 }}>Scope</div>
                <div style={{ width: 24 }} />
              </div>
            )}
            {fields.map(({ key, name, ...restField }) => (
              <Space
                key={key}
                className="flex w-full"
                align="baseline"
                size="middle"
              >
                <Form.Item
                  {...restField}
                  name={[name, "name"]}
                  className="flex-1 mb-0"
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
                <Form.Item
                  {...restField}
                  name={[name, "value"]}
                  className="flex-1 mb-0"
                  shouldUpdate
                >
                  <ValueField fieldName={name} />
                </Form.Item>
                <Form.Item
                  {...restField}
                  name={[name, "scope"]}
                  className="mb-0"
                  initialValue="global"
                  style={{ width: 160 }}
                >
                  <Select
                    options={[
                      { value: "global", label: "Global" },
                      { value: "per_user", label: "Per-user field" },
                    ]}
                  />
                </Form.Item>
                <MinusCircleOutlined
                  onClick={() => remove(name)}
                  className="text-gray-500 hover:text-red-500 cursor-pointer"
                />
              </Space>
            ))}
            <Button
              type="dashed"
              onClick={() => add({ scope: "global" })}
              icon={<PlusOutlined />}
              block
            >
              Add Environment Variable
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
        isPerUser ? "set by each user" : "e.g. postgresql"
      }
      disabled={isPerUser}
      className="rounded-md font-mono"
    />
  );
};

export default EnvVarsSection;
