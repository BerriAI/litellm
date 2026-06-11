import React from "react";
import { Alert, Form, Switch, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const DelegateAuthToUpstreamField: React.FC = () => {
  const form = Form.useFormInstance();
  const delegateAuth = Form.useWatch("delegate_auth_to_upstream", form) === true;
  const internalOnly = Form.useWatch("available_on_public_internet", form) === false;

  return (
    <div className="space-y-3 mb-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <span className="text-sm font-medium text-gray-700 flex items-center">
            Delegate auth to upstream (PKCE passthrough)
            <Tooltip title="When on, LiteLLM skips its own API key/SSO check for this server and lets the client complete PKCE directly with the upstream MCP server. Only honored for the Interactive (PKCE) flow. No spend tracking or per-key rate limiting will run on this route.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
          <p className="text-sm text-gray-600 mt-1">
            Bypass LiteLLM auth so clients authenticate directly with the upstream OAuth MCP server.
          </p>
        </div>
        <Form.Item name="delegate_auth_to_upstream" valuePropName="checked" className="mb-0">
          <Switch />
        </Form.Item>
      </div>

      {delegateAuth && (
        <div className="p-3 bg-blue-50 rounded-lg text-sm text-blue-700 flex items-start gap-2">
          <InfoCircleOutlined className="mt-0.5 flex-shrink-0" />
          <span>
            Clients authenticate directly with the upstream MCP server. LiteLLM won&apos;t require its own API key/SSO
            on this route and won&apos;t store user credentials, so this endpoint is reachable without a LiteLLM login.
          </span>
        </div>
      )}

      {delegateAuth && internalOnly && (
        <Alert
          type="warning"
          showIcon
          message="Internal server with upstream OAuth delegation"
          description="This MCP server is configured as internal-only but delegates auth to upstream. Anonymous users will be able to reach the upstream OAuth2 /authorize flow without a LiteLLM session. Ensure your upstream provider and network enforce access controls."
        />
      )}
    </div>
  );
};

export default DelegateAuthToUpstreamField;
