import React from "react";
import { Form, Input, Select, Button, Tooltip, Typography } from "antd";
import {
  InfoCircleOutlined,
  MinusCircleOutlined,
  PlusOutlined,
} from "@ant-design/icons";

const { Text } = Typography;

const SCOPE_OPTIONS = [
  { value: "global", label: "Instance" },
  { value: "user", label: "Per-user" },
];

/**
 * Form section for admin-configured MCP variables.
 *
 * Each row has: name | value | scope. Variables can be interpolated into
 * Static Headers via ${NAME}. ``scope=global`` (shown as "Instance") values
 * are used as-is. ``scope=user`` (shown as "Per-user") values are filled in
 * by each user via the MCP Gateway dashboard.
 *
 * The parent form reads the ``variables`` field from the form values.
 */
const VariablesSection: React.FC = () => {
  return (
    <div className="rounded-lg border border-dashed border-purple-300 bg-purple-50 p-4">
      <div className="flex items-center gap-2 mb-1">
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
              credentials) via the MCP Gateway dashboard.
            </>
          }
        >
          <InfoCircleOutlined className="text-purple-500" />
        </Tooltip>
      </div>
      <Text className="text-xs text-gray-600 block mb-3">
        Reference these in Static Headers or Authentication as{" "}
        <code>{"${VAR_NAME}"}</code>. For example:{" "}
        <code className="bg-white px-1 rounded border border-gray-200">
          {"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}"}
        </code>
      </Text>

      <Form.List name="variables">
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
              <div key={key} className="flex gap-3 items-baseline">
                <Form.Item
                  {...restField}
                  name={[name, "name"]}
                  className="mb-0"
                  style={{ flex: 1 }}
                  rules={[
                    { required: true, message: "Variable name is required" },
                    {
                      pattern: /^[A-Za-z_][A-Za-z0-9_]*$/,
                      message: "Use letters, digits, underscores; cannot start with a digit.",
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
                  className="mb-0"
                  style={{ flex: 1 }}
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
                  <Select options={SCOPE_OPTIONS} />
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
              onClick={() => add({ scope: "global" })}
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

// Disables the value field when scope=user (those values come from each
// user later), keeping the column visible so the row layout stays consistent.
const ValueField: React.FC<{
  fieldName: number;
  value?: string;
  onChange?: (v: string) => void;
}> = ({ fieldName, value, onChange }) => {
  const scope = Form.useWatch(["variables", fieldName, "scope"]);
  const isPerUser = scope === "user";
  return (
    <Input
      value={value ?? ""}
      onChange={(e) => onChange?.(e.target.value)}
      placeholder={isPerUser ? "Defined per user" : "e.g. postgresql"}
      disabled={isPerUser}
      className="rounded-md font-mono"
    />
  );
};

export default VariablesSection;
