import React from "react";
import { Form, Input, Button, Tooltip, Radio, Typography } from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";

const { Text: AntdText } = Typography;

const MCPHeaderVariablesSection: React.FC = () => (
  <Form.Item
    label={
      <span className="text-sm font-medium text-gray-700 flex items-center">
        Header Variables
        <Tooltip title="Define variables that can be interpolated into the Static Headers above using ${VARIABLE_NAME} syntax. Global variables are shared across all users. Per-user variables are filled in by each user from their own dashboard.">
          <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
        </Tooltip>
      </span>
    }
    required={false}
  >
    <Form.List name="header_variables">
      {(fields, { add, remove }) => (
        <div className="space-y-3">
          {fields.length > 0 && (
            <div className="grid grid-cols-12 gap-3 px-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
              <div className="col-span-4">Variable Name</div>
              <div className="col-span-4">Value</div>
              <div className="col-span-3">Scope</div>
              <div className="col-span-1" />
            </div>
          )}
          {fields.map(({ key, name, ...restField }) => (
            <Form.Item
              shouldUpdate={(prev, cur) =>
                prev?.header_variables?.[name]?.scope !== cur?.header_variables?.[name]?.scope
              }
              noStyle
              key={key}
            >
              {({ getFieldValue }) => {
                const scope = getFieldValue(["header_variables", name, "scope"]) ?? "global";
                const isPerUser = scope === "per_user";
                return (
                  <div className="grid grid-cols-12 gap-3 items-start">
                    <Form.Item
                      {...restField}
                      name={[name, "name"]}
                      className="col-span-4 mb-0"
                      rules={[
                        { required: true, message: "Required" },
                        {
                          pattern: /^[A-Za-z_][A-Za-z0-9_]*$/,
                          message: "Letters, digits, underscore. No spaces.",
                        },
                      ]}
                    >
                      <Input size="large" allowClear placeholder="e.g., DB_PROTOCOL" className="font-mono" />
                    </Form.Item>
                    <Form.Item
                      {...restField}
                      name={[name, "value"]}
                      className="col-span-4 mb-0"
                    >
                      <Input
                        size="large"
                        allowClear
                        disabled={isPerUser}
                        placeholder={isPerUser ? "Filled in by each user" : "Value"}
                      />
                    </Form.Item>
                    <Form.Item
                      {...restField}
                      name={[name, "scope"]}
                      className="col-span-3 mb-0"
                      initialValue="global"
                    >
                      <Radio.Group size="middle" buttonStyle="solid">
                        <Radio.Button value="global">Global</Radio.Button>
                        <Radio.Button value="per_user">Per-user</Radio.Button>
                      </Radio.Group>
                    </Form.Item>
                    <div className="col-span-1 flex justify-center pt-3">
                      <MinusCircleOutlined
                        onClick={() => remove(name)}
                        className="text-gray-500 hover:text-red-500 cursor-pointer"
                      />
                    </div>
                  </div>
                );
              }}
            </Form.Item>
          ))}
          <Button
            type="dashed"
            onClick={() => add({ scope: "global" })}
            icon={<PlusOutlined />}
            block
          >
            Add Variable
          </Button>
          <AntdText type="secondary" className="text-xs block mt-2">
            Reference these in the Static Headers above as <code>{"${VARIABLE_NAME}"}</code>. Example:{" "}
            <code className="font-mono">{"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOST}"}</code>
          </AntdText>
        </div>
      )}
    </Form.List>
  </Form.Item>
);

export default MCPHeaderVariablesSection;
