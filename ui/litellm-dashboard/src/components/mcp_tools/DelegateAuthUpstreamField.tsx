import React from "react";
import { Alert, Form, Switch, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

interface DelegateAuthUpstreamFieldProps {
  initialValue?: boolean;
}

const DelegateAuthUpstreamField: React.FC<DelegateAuthUpstreamFieldProps> = ({ initialValue = false }) => {
  const form = Form.useFormInstance();
  const watchedDelegateAuth = Form.useWatch("delegate_auth_to_upstream", form);
  const watchedPublicInternet = Form.useWatch("available_on_public_internet", form);
  const showInternalWarning = watchedDelegateAuth === true && watchedPublicInternet === false;

  return (
    <>
      <div className="flex items-start justify-between gap-4">
        <div>
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Delegate auth to upstream (PKCE passthrough)
            <Tooltip title="When on, LiteLLM skips its own API key/SSO check for this server and lets the client complete PKCE directly with the upstream MCP server. No spend tracking or per-key rate limiting will run on this route.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
          <p className="text-sm text-gray-600 mt-1">
            Bypass LiteLLM auth so clients authenticate directly with the upstream OAuth MCP server.
          </p>
        </div>
        <Form.Item
          name="delegate_auth_to_upstream"
          valuePropName="checked"
          initialValue={initialValue}
          className="mb-0"
        >
          <Switch />
        </Form.Item>
      </div>

      {showInternalWarning && (
        <Alert
          type="warning"
          showIcon
          className="mt-3"
          message="Internal server with upstream OAuth delegation"
          description="This MCP server is configured as internal-only but delegates auth to upstream. Anonymous users will be able to reach the upstream OAuth2 /authorize flow without a LiteLLM session. Ensure your upstream provider and network enforce access controls."
        />
      )}
    </>
  );
};

export default DelegateAuthUpstreamField;
