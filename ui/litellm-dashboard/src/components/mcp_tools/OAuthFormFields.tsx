import React from "react";
import { Form, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { OAUTH_FLOW } from "./types";

interface OAuthFlowStatus {
  startOAuthFlow: () => void;
  status: string;
  error: string | null;
  tokenResponse: { access_token?: string; expires_in?: number } | null;
}

interface OAuthFormFieldsProps {
  isM2M: boolean;
  isEditing?: boolean;
  oauthFlow?: OAuthFlowStatus;
  initialFlowType?: string;
}

const fieldClassName = "rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-gray-700 flex items-center">
    {label}
    <Tooltip title={tooltip}>
      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
    </Tooltip>
  </span>
);

const OAuthFormFields: React.FC<OAuthFormFieldsProps> = ({
  isM2M,
  isEditing = false,
  oauthFlow,
  initialFlowType,
}) => {
  const placeholderSuffix = isEditing ? " (leave blank to keep existing)" : "";

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label="OAuth Flow Type"
            tooltip="Choose how the proxy authenticates with this MCP server. M2M is for server-to-server communication using client credentials. Interactive (PKCE) is for user-facing flows that require browser-based authorization."
          />
        }
        name="oauth_flow_type"
        {...(initialFlowType ? { initialValue: initialFlowType } : {})}
      >
        <Select className="rounded-lg" size="large">
          <Select.Option value={OAUTH_FLOW.M2M}>
            <div>
              <span className="font-medium">Machine-to-Machine (M2M)</span>
              <span className="text-gray-400 text-xs ml-2">server-to-server, no user interaction</span>
            </div>
          </Select.Option>
          <Select.Option value={OAUTH_FLOW.INTERACTIVE}>
            <div>
              <span className="font-medium">Interactive (PKCE)</span>
              <span className="text-gray-400 text-xs ml-2">browser-based user authorization</span>
            </div>
          </Select.Option>
        </Select>
      </Form.Item>

      {isM2M ? (
        <>
          <Form.Item
            label={<FieldLabel label="Client ID" tooltip="OAuth2 client ID for the client_credentials grant." />}
            name={["credentials", "client_id"]}
            rules={[{ required: true, message: "Client ID is required for M2M OAuth" }]}
          >
            <TextInput type="password" placeholder={`Enter OAuth client ID${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Client Secret" tooltip="OAuth2 client secret for the client_credentials grant." />}
            name={["credentials", "client_secret"]}
            rules={[{ required: true, message: "Client Secret is required for M2M OAuth" }]}
          >
            <TextInput type="password" placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Token URL" tooltip="Token endpoint URL for the client_credentials grant." />}
            name="token_url"
            rules={[{ required: true, message: "Token URL is required for M2M OAuth" }]}
          >
            <TextInput placeholder="https://auth.example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes to request with the client_credentials grant." />}
            name={["credentials", "scopes"]}
          >
            <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
          </Form.Item>
        </>
      ) : (
        <>
          <Form.Item
            label={<FieldLabel label="Client ID (optional)" tooltip="Provide only if your MCP server cannot handle dynamic client registration." />}
            name={["credentials", "client_id"]}
          >
            <TextInput type="password" placeholder={`Enter client ID${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Client Secret (optional)" tooltip="Provide only if your MCP server cannot handle dynamic client registration." />}
            name={["credentials", "client_secret"]}
          >
            <TextInput type="password" placeholder={`Enter client secret${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes requested during token exchange. Separate multiple scopes with enter or commas." />}
            name={["credentials", "scopes"]}
          >
            <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Authorization URL (optional)" tooltip="Optional override for the authorization endpoint." />}
            name="authorization_url"
          >
            <TextInput placeholder="https://example.com/oauth/authorize" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Token URL (optional)" tooltip="Optional override for the token endpoint." />}
            name="token_url"
          >
            <TextInput placeholder="https://example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Registration URL (optional)" tooltip="Optional override for the dynamic client registration endpoint." />}
            name="registration_url"
          >
            <TextInput placeholder="https://example.com/oauth/register" className={fieldClassName} />
          </Form.Item>
          {oauthFlow && (
            <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2">
              <p className="text-sm text-gray-600">
                Use OAuth to fetch a fresh access token and temporarily save it in the session as the authentication value.
              </p>
              <Button
                variant="secondary"
                onClick={oauthFlow.startOAuthFlow}
                disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
              >
                {oauthFlow.status === "authorizing"
                  ? "Waiting for authorization..."
                  : oauthFlow.status === "exchanging"
                    ? "Exchanging authorization code..."
                    : "Authorize & Fetch Token"}
              </Button>
              {oauthFlow.error && <p className="text-sm text-red-500">{oauthFlow.error}</p>}
              {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
                <p className="text-sm text-green-600">
                  Token fetched. Expires in {oauthFlow.tokenResponse.expires_in ?? "?"} seconds.
                </p>
              )}
            </div>
          )}
        </>
      )}
    </>
  );
};

export default OAuthFormFields;
