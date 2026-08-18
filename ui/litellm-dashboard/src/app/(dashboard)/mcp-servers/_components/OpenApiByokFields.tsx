import React from "react";
import { Form, Input, Select, Switch, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const OpenApiByokFields: React.FC = () => (
  <>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
          BYOK (Bring Your Own Key)
          <Tooltip title="When enabled, each user provides their own API key for this service. Keys are stored per-user and never shared.">
            <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name="is_byok"
      valuePropName="checked"
    >
      <Switch />
    </Form.Item>

    <Form.Item noStyle shouldUpdate={(prev, cur) => prev.is_byok !== cur.is_byok || prev.auth_type !== cur.auth_type}>
      {({ getFieldValue }) =>
        getFieldValue("is_byok") ? (
          <>
            {/* Auth format hint */}
            {getFieldValue("auth_type") && getFieldValue("auth_type") !== "none" && (
              <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700 flex items-start gap-2">
                <InfoCircleOutlined className="mt-0.5 shrink-0" />
                <span>
                  User keys will be sent as:{" "}
                  <code className="font-mono bg-blue-100 px-1 rounded-sm">
                    {getFieldValue("auth_type") === "bearer_token" && "Authorization: Bearer {key}"}
                    {getFieldValue("auth_type") === "token" && "Authorization: token {key}"}
                    {getFieldValue("auth_type") === "api_key" && "x-api-key: {key}"}
                    {getFieldValue("auth_type") === "basic" && "Authorization: Basic {key}"}
                    {getFieldValue("auth_type") === "authorization" && "Authorization: {key}"}
                  </code>
                  {!getFieldValue("auth_type") && "Set Authentication Type below to specify the format."}
                </span>
              </div>
            )}
            {!getFieldValue("auth_type") && (
              <div className="mb-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700 flex items-start gap-2">
                <InfoCircleOutlined className="mt-0.5 shrink-0" />
                <span>
                  Set the <strong>Authentication Type</strong> below to specify how user keys are sent (e.g., Bearer
                  Token, API Key header).
                </span>
              </div>
            )}
            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700">
                  Access Description
                  <Tooltip title="List of permissions shown to users in the connection modal (e.g. 'Create and manage Jira issues')">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="byok_description"
            >
              <Select
                mode="tags"
                placeholder="Add access description items (press Enter after each)"
                className="w-full"
                tokenSeparators={[","]}
              />
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700">
                  API Key Help URL
                  <Tooltip title="Optional link shown to users to help them find their API key">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="byok_api_key_help_url"
            >
              <Input placeholder="https://docs.example.com/api-keys" />
            </Form.Item>
          </>
        ) : null
      }
    </Form.Item>
  </>
);

export default OpenApiByokFields;
