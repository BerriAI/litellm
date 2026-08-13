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
            label="Org Token Endpoint (leg 1)"
            tooltip="Your IdP org authorization server's token endpoint. LiteLLM exchanges the user's identity assertion here for an ID-JAG assertion (RFC 8693 with requested_token_type=urn:ietf:params:oauth:token-type:id-jag)."
          />
        }
        name="token_exchange_endpoint"
        rules={[{ required: !isEditing, message: "The org token endpoint is required for ID-JAG" }]}
      >
        <Input placeholder="https://your-org.okta.com/oauth2/v1/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Resource Token Endpoint (leg 2)"
            tooltip="The upstream resource authorization server's token endpoint. LiteLLM posts the ID-JAG assertion here as an RFC 7523 jwt-bearer grant to get the access token the MCP server accepts."
          />
        }
        name={["credentials", "id_jag_resource_token_endpoint"]}
        rules={[{ required: !isEditing, message: "The resource token endpoint is required for ID-JAG" }]}
      >
        <Input placeholder="https://upstream.example.com/oauth2/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={<FieldLabel label="Client ID" tooltip="OAuth2 client ID LiteLLM authenticates as on both legs." />}
        name={["credentials", "client_id"]}
        rules={[{ required: !isEditing, message: "Client ID is required for ID-JAG" }]}
      >
        <Input.Password placeholder={`Enter OAuth client ID${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client Secret"
            tooltip="Authenticates LiteLLM as the OAuth client via client_secret_post. Leave blank when using a private key instead; a private key takes precedence over this secret."
          />
        }
        name={["credentials", "client_secret"]}
        dependencies={[["credentials", "client_private_key"]]}
        rules={[
          ({ getFieldValue }) => ({
            validator: (_, value) => {
              if (isEditing || value || getFieldValue(["credentials", "client_private_key"])) {
                return Promise.resolve();
              }
              return Promise.reject(new Error("Provide either a client secret or a client private key"));
            },
          }),
        ]}
      >
        <Input.Password placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client Private Key (PEM)"
            tooltip="PEM private key signing the RFC 7523 private_key_jwt client assertion. Okta Cross App Access normally requires this. When set it takes precedence over the client secret."
          />
        }
        name={["credentials", "client_private_key"]}
      >
        <Input.TextArea
          rows={3}
          placeholder={`-----BEGIN PRIVATE KEY-----${placeholderSuffix}`}
          className={fieldClassName}
        />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Private Key ID (optional)"
            tooltip="The kid advertised in the client assertion JWT header, so the IdP can select the right registered key."
          />
        }
        name={["credentials", "client_private_key_id"]}
      >
        <Input placeholder="my-signing-key-1" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client Assertion Signing Algorithm (optional)"
            tooltip="Algorithm signing the client assertion JWT. Defaults to RS256."
          />
        }
        name={["credentials", "client_assertion_signing_alg"]}
      >
        <Input placeholder="RS256" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Audience (optional)"
            tooltip="RFC 8693 audience sent on leg 1, identifying the upstream the ID-JAG assertion is minted for."
          />
        }
        name="audience"
      >
        <Input placeholder="https://upstream.example.com" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Resource Indicator (optional)"
            tooltip="RFC 8707 resource indicator sent on leg 1. Separate from Audience, which is the RFC 8693 parameter."
          />
        }
        name={["credentials", "id_jag_resource"]}
      >
        <Input placeholder="https://upstream.example.com/mcp" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Subject Token Type (optional)"
            tooltip="Type of the identity assertion exchanged on leg 1. Defaults to urn:ietf:params:oauth:token-type:id_token."
          />
        }
        name="subject_token_type"
      >
        <Input placeholder="urn:ietf:params:oauth:token-type:id_token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={<FieldLabel label="Scopes (optional)" tooltip="Scopes requested on leg 1 of the exchange." />}
        name={["credentials", "scopes"]}
      >
        <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
      </Form.Item>
    </>
  );
};

export default IdJagFormFields;
