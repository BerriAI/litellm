import React from "react";
import { Form, Select, Input, Switch, Tooltip } from "antd";
import type { FormInstance } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { AUTH_TYPE } from "./types";

export const BYOK_AUTH_FORMAT_HINT: Record<string, string> = {
  [AUTH_TYPE.BEARER_TOKEN]: "Authorization: Bearer {key}",
  [AUTH_TYPE.TOKEN]: "Authorization: token {key}",
  [AUTH_TYPE.API_KEY]: "x-api-key: {key}",
  [AUTH_TYPE.BASIC]: "Authorization: Basic {key}",
};

interface ByokFieldsProps {
  form: FormInstance;
}

const ByokFields: React.FC<ByokFieldsProps> = ({ form }) => {
  const isByok = Form.useWatch("is_byok", form);
  const authType = Form.useWatch("auth_type", form) as string | undefined;
  const formatHint = authType ? BYOK_AUTH_FORMAT_HINT[authType] : undefined;

  return (
    <>
      <Form.Item
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
            BYOK (Bring Your Own Key)
            <Tooltip title="When enabled, each user supplies their own API key for this server instead of a single shared key. Keys are stored per-user and never shared.">
              <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        name="is_byok"
        valuePropName="checked"
      >
        <Switch />
      </Form.Item>

      {isByok && (
        <>
          {formatHint && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700 flex items-start gap-2">
              <InfoCircleOutlined className="mt-0.5 flex-shrink-0" />
              <span>
                User keys will be sent as: <code className="font-mono bg-blue-100 px-1 rounded">{formatHint}</code>
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
      )}
    </>
  );
};

export default ByokFields;
