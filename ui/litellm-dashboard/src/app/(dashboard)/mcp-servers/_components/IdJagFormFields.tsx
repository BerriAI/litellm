import React, { useState } from "react";
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
  const [clientAuthMethod, setClientAuthMethod] = useState<"client_secret" | "private_key_jwt">("client_secret");

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label="IdP Token Exchange Endpoint"
            tooltip="Your enterprise identity provider's token endpoint (the organization authorization server). The gateway exchanges the user's SSO identity assertion here for an ID-JAG (RFC 8693 with requested_token_type id-jag)."
          />
        }
        name="token_exchange_endpoint"
        rules={[{ required: !isEditing, message: "The IdP token exchange endpoint is required for ID-JAG" }]}
      >
        <Input placeholder="https://your-org.okta.com/oauth2/v1/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Resource Token Endpoint (optional)"
            tooltip="The upstream MCP server's authorization server token endpoint, where the ID-JAG is redeemed for the access token (RFC 7523 jwt-bearer). Leave blank to auto-discover it from the upstream's protected-resource metadata (RFC 9728 then RFC 8414); discovery only trusts an authorization server that advertises the id-jag grant profile."
          />
        }
        name={["credentials", "id_jag_resource_token_endpoint"]}
      >
        <Input placeholder="https://mcp-as.example.com/oauth2/token" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={
          <FieldLabel
            label="Client ID"
            tooltip="The gateway's OAuth client ID. It must be registered at BOTH the identity provider and the upstream's authorization server, and the ID-JAG's client_id claim must match it."
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
            label="Client Authentication"
            tooltip="How the gateway authenticates to both token endpoints: a shared client secret, or a private key signing a JWT client assertion (private_key_jwt)."
          />
        }
      >
        <Select<"client_secret" | "private_key_jwt">
          className="rounded-lg"
          size="large"
          value={clientAuthMethod}
          onChange={setClientAuthMethod}
          options={[
            { value: "client_secret", label: <span className="font-medium">Client Secret</span> },
            { value: "private_key_jwt", label: <span className="font-medium">Private Key JWT</span> },
          ]}
        />
      </Form.Item>
      {clientAuthMethod === "client_secret" && (
        <Form.Item
          label={
            <FieldLabel
              label="Client Secret"
              tooltip="OAuth2 client secret used to authenticate to both token endpoints."
            />
          }
          name={["credentials", "client_secret"]}
          rules={[{ required: !isEditing, message: "Client Secret is required for ID-JAG" }]}
        >
          <Input.Password placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
        </Form.Item>
      )}
      {clientAuthMethod === "private_key_jwt" && (
        <>
          <Form.Item
            label={
              <FieldLabel
                label="Client Private Key"
                tooltip="PEM private key used to sign the JWT client assertion (private_key_jwt). Stored encrypted at rest."
              />
            }
            name={["credentials", "client_private_key"]}
            rules={[{ required: !isEditing, message: "A private key is required for private_key_jwt" }]}
          >
            <Input.TextArea
              rows={4}
              placeholder={`-----BEGIN PRIVATE KEY-----${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Key ID (optional)"
                tooltip="The kid header for the client assertion, when the authorization server needs one to select the verification key."
              />
            }
            name={["credentials", "client_private_key_id"]}
          >
            <Input placeholder="key-2026-01" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Signing Algorithm (optional)"
                tooltip="JWS algorithm for the client assertion. Defaults to RS256."
              />
            }
            name={["credentials", "client_assertion_signing_alg"]}
          >
            <Input placeholder="RS256" className={fieldClassName} />
          </Form.Item>
        </>
      )}
      <Form.Item
        label={
          <FieldLabel
            label="Audience (optional)"
            tooltip="The upstream authorization server's identifier, sent as the RFC 8693 audience so the IdP mints the ID-JAG for it."
          />
        }
        name="audience"
      >
        <Input placeholder="https://mcp-as.example.com" className={fieldClassName} />
      </Form.Item>
      <Form.Item
        label={<FieldLabel label="Scopes (optional)" tooltip="Scopes requested in the ID-JAG exchange." />}
        name={["credentials", "scopes"]}
      >
        <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
      </Form.Item>
    </>
  );
};

export default IdJagFormFields;
