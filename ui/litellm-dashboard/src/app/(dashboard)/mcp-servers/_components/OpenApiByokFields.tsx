import React from "react";
import { Input, Select, Switch, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useWatch } from "react-hook-form";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { selectControl, switchControl, textControl } from "./mcpFieldRules";

const AUTH_HEADER_FORMATS: Readonly<Record<string, string>> = {
  bearer_token: "Authorization: Bearer {key}",
  token: "Authorization: token {key}",
  api_key: "x-api-key: {key}",
  basic: "Authorization: Basic {key}",
  authorization: "Authorization: {key}",
};

const OpenApiByokFields: React.FC = () => {
  const isByok = Boolean(useWatch({ name: "is_byok" }));
  const authType = useWatch({ name: "auth_type" }) as string | undefined;
  const hasAuthType = Boolean(authType) && authType !== "none";

  return (
    <>
      <MountedFormField
        label={
          <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
            BYOK (Bring Your Own Key)
            <Tooltip title="When enabled, each user provides their own API key for this service. Keys are stored per-user and never shared.">
              <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        name="is_byok"
      >
        {(control) => <Switch {...switchControl(control)} />}
      </MountedFormField>

      {isByok && (
        <>
          {hasAuthType && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700 flex items-start gap-2">
              <InfoCircleOutlined className="mt-0.5 shrink-0" />
              <span>
                User keys will be sent as:{" "}
                <code className="font-mono bg-blue-100 px-1 rounded-sm">
                  {authType === undefined ? "" : AUTH_HEADER_FORMATS[authType]}
                </code>
              </span>
            </div>
          )}
          {!authType && (
            <div className="mb-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700 flex items-start gap-2">
              <InfoCircleOutlined className="mt-0.5 shrink-0" />
              <span>
                Set the <strong>Authentication Type</strong> below to specify how user keys are sent (e.g., Bearer
                Token, API Key header).
              </span>
            </div>
          )}
          <MountedFormField
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
            {(control) => (
              <Select
                {...selectControl(control)}
                mode="tags"
                placeholder="Add access description items (press Enter after each)"
                className="w-full"
                tokenSeparators={[","]}
              />
            )}
          </MountedFormField>

          <MountedFormField
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
            {(control) => <Input {...textControl(control)} placeholder="https://docs.example.com/api-keys" />}
          </MountedFormField>
        </>
      )}
    </>
  );
};

export default OpenApiByokFields;
