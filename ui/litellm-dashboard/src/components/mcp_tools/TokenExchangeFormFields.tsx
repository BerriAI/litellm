import React from "react";
import { Form, Input, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

interface TokenExchangeFormFieldsProps {
  isEditing?: boolean;
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

const TokenExchangeFormFields: React.FC<TokenExchangeFormFieldsProps> = ({ isEditing = false }) => {
  const placeholderSuffix = isEditing ? " (leave blank to keep existing)" : "";

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label="Token Exchange Endpoint (optional)"
            tooltip="RFC 8693 token endpoint. The proxy exchanges the user's incoming token here for a scoped token used to call the upstream MCP server. Leave blank to auto-discover it from the upstream's protected-resource metadata (RFC 9728 then RFC 8414)."
          />
        }
        name="token_exchange_endpoint"
      >
        <Input placeholder="https://idp.example.com/oauth2/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client ID"
            tooltip="OAuth2 client ID used to authenticate to the token exchange endpoint."
          />
        }
        name={["credentials", "client_id"]}
        rules={[{ required: !isEditing, message: "Client ID is required for token exchange" }]}
      >
        <Input.Password placeholder={`Enter OAuth client ID${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client Secret"
            tooltip="OAuth2 client secret used to authenticate to the token exchange endpoint."
          />
        }
        name={["credentials", "client_secret"]}
        rules={[{ required: !isEditing, message: "Client Secret is required for token exchange" }]}
      >
        <Input.Password placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Audience (optional)"
            tooltip="Target audience for the exchanged token (RFC 8693 audience). Identifies the upstream MCP server the token is for."
          />
        }
        name="audience"
      >
        <Input placeholder="https://upstream.example.com" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Subject Token Type (optional)"
            tooltip="Type of the user's incoming token (RFC 8693 subject_token_type). Defaults to urn:ietf:params:oauth:token-type:access_token."
          />
        }
        name="subject_token_type"
      >
        <Input placeholder="urn:ietf:params:oauth:token-type:access_token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes to request during the token exchange." />}
        name={["credentials", "scopes"]}
      >
        <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
      </Form.Item>
    </>
  );
};

export default TokenExchangeFormFields;
