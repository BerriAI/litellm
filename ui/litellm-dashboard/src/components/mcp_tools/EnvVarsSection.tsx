import React from "react";
import { Form, Input, Select, Space, Button, Tooltip, Typography } from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";

const { Text } = Typography;

const SCOPE_OPTIONS = [
  { value: "global", label: "Global" },
  { value: "user", label: "Per-user" },
];

/**
 * Form section for admin-configured MCP environment variables.
 *
 * Each row has: name | value | scope. Variables can be interpolated into
 * Static Headers via ${NAME}. ``scope=global`` values are used as-is.
 * ``scope=user`` values are filled in by each user — the admin-entered
 * value is just a placeholder/description.
 *
 * The parent form must render this inside a ``<Form>`` and read the
 * ``env_vars`` field from the form values.
 */
const EnvVarsSection: React.FC = () => {
  return (
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          Environment Variables
          <Tooltip
            title={
              <div>
                <div>
                  Define variables that get interpolated into Static Headers via{" "}
                  <code>{"${NAME}"}</code> syntax.
                </div>
                <div className="mt-2">
                  <b>Global</b>: value is used for every user.
                </div>
                <div>
                  <b>Per-user</b>: each user fills in their own value via the
                  MCP Gateway dashboard. The value you enter here is shown to
                  the user as a placeholder/description.
                </div>
              </div>
            }
          >
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      required={false}
    >
      <Text className="block text-xs text-gray-500 mb-2">
        Reference these in Static Headers like{" "}
        <code className="bg-gray-100 px-1 rounded">{"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@..."}</code>
      </Text>
      <Form.List name="env_vars">
        {(fields, { add, remove }) => (
          <div className="space-y-3">
            {fields.map(({ key, name, ...restField }) => (
              <Space key={key} className="flex w-full" align="baseline" size="middle">
                <Form.Item
                  {...restField}
                  name={[name, "name"]}
                  className="flex-1"
                  rules={[
                    { required: true, message: "Variable name is required" },
                    {
                      pattern: /^[A-Za-z_][A-Za-z0-9_]*$/,
                      message: "Use letters, digits, underscores; cannot start with a digit.",
                    },
                  ]}
                >
                  <Input size="large" allowClear className="rounded-lg" placeholder="DB_PROTOCOL" />
                </Form.Item>
                <Form.Item
                  {...restField}
                  name={[name, "value"]}
                  className="flex-1"
                >
                  <Input
                    size="large"
                    allowClear
                    className="rounded-lg"
                    placeholder="postgres / placeholder for user-scoped"
                  />
                </Form.Item>
                <Form.Item
                  {...restField}
                  name={[name, "scope"]}
                  className="w-36"
                  initialValue="global"
                  rules={[{ required: true, message: "Scope required" }]}
                >
                  <Select size="large" options={SCOPE_OPTIONS} />
                </Form.Item>
                <MinusCircleOutlined
                  onClick={() => remove(name)}
                  className="text-gray-500 hover:text-red-500 cursor-pointer"
                />
              </Space>
            ))}
            <Button
              type="dashed"
              onClick={() => add({ name: "", value: "", scope: "global" })}
              icon={<PlusOutlined />}
              block
            >
              Add Environment Variable
            </Button>
          </div>
        )}
      </Form.List>
    </Form.Item>
  );
};

export default EnvVarsSection;
