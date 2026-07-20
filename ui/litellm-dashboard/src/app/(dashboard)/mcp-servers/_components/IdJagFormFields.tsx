import React from "react";
import { Form, Input, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

interface IdJagFormFieldsProps {
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

const IdJagFormFields: React.FC<IdJagFormFieldsProps> = ({ isEditing = false }) => {
  const placeholderSuffix = isEditing ? " (leave blank to keep existing)" : "";

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label="IdP Token Endpoint (leg 1)"
            tooltip="The identity provider's org token endpoint where the gateway exchanges the caller's identity token for an ID-JAG assertion (RFC 8693 token-exchange). For Okta this is https://<your-okta-domain>/oauth2/v1/token."
          />
        }
        name="token_exchange_endpoint"
        rules={[{ required: !isEditing, message: "The IdP token endpoint is required for ID-JAG" }]}
      >
        <Input placeholder="https://your-okta-domain.okta.com/oauth2/v1/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Resource Token Endpoint (leg 2)"
            tooltip="The resource app's own token endpoint where the gateway presents the ID-JAG assertion (RFC 7523 jwt-bearer) to obtain a scoped access token for the upstream MCP server. This is the resource server's authorization server, not the IdP."
          />
        }
        name={["credentials", "id_jag_resource_token_endpoint"]}
        rules={[{ required: !isEditing, message: "The resource token endpoint is required for ID-JAG" }]}
      >
        <Input placeholder="https://mcp.example.com/oauth2/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client ID"
            tooltip="OAuth2 client ID the gateway (the requesting app registered with your IdP) uses to authenticate on both legs of the exchange."
          />
        }
        name={["credentials", "client_id"]}
        rules={[{ required: !isEditing, message: "Client ID is required for ID-JAG" }]}
      >
        <Input.Password placeholder={`Enter OAuth client ID${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client Secret"
            tooltip="OAuth2 client secret used to authenticate the gateway to the token endpoints."
          />
        }
        name={["credentials", "client_secret"]}
        rules={[{ required: !isEditing, message: "Client Secret is required for ID-JAG" }]}
      >
        <Input.Password placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Audience (optional)"
            tooltip="Target audience sent on leg 1 (RFC 8693 audience). Identifies the resource app the ID-JAG assertion is minted for."
          />
        }
        name="audience"
      >
        <Input placeholder="api://your-mcp-resource" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Resource Indicator (optional)"
            tooltip="RFC 8707 resource indicator sent on leg 1, identifying the protected resource the token is intended for."
          />
        }
        name={["credentials", "id_jag_resource"]}
      >
        <Input placeholder="https://mcp.example.com" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes requested on leg 1 of the exchange." />}
        name={["credentials", "scopes"]}
      >
        <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
      </Form.Item>
    </>
  );
};

export default IdJagFormFields;
