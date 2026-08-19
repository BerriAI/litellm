import React from "react";
import { Input, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useFormContext } from "react-hook-form";
import { MountedFormField, bindControl, type MountedFormValues } from "@/components/common_components/MountedFormField";

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
  const { getValues } = useFormContext<MountedFormValues>();

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label="Org Token Endpoint (leg 1)"
            tooltip="Your IdP org authorization server's token endpoint. LiteLLM exchanges the user's identity assertion here for an ID-JAG assertion (RFC 8693 with requested_token_type=urn:ietf:params:oauth:token-type:id-jag)."
          />
        }
        name="token_exchange_endpoint"
        required={!isEditing}
        rules={isEditing ? {} : { required: "The org token endpoint is required for ID-JAG" }}
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="https://your-org.okta.com/oauth2/v1/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Resource Token Endpoint (leg 2)"
            tooltip="The upstream resource authorization server's token endpoint. LiteLLM posts the ID-JAG assertion here as an RFC 7523 jwt-bearer grant to get the access token the MCP server accepts."
          />
        }
        name="credentials.id_jag_resource_token_endpoint"
        required={!isEditing}
        rules={isEditing ? {} : { required: "The resource token endpoint is required for ID-JAG" }}
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="https://upstream.example.com/oauth2/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={<FieldLabel label="Client ID" tooltip="OAuth2 client ID LiteLLM authenticates as on both legs." />}
        name="credentials.client_id"
        required={!isEditing}
        rules={isEditing ? {} : { required: "Client ID is required for ID-JAG" }}
      >
        {(field) => (
          <Input.Password
            {...bindControl<string | undefined>(field)}
            placeholder={`Enter OAuth client ID${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Client Secret"
            tooltip="Authenticates LiteLLM as the OAuth client via client_secret_post. Leave blank when using a private key instead; a private key takes precedence over this secret."
          />
        }
        name="credentials.client_secret"
        rules={{
          deps: ["credentials.client_private_key"],
          validate: (value) =>
            isEditing || value || getValues("credentials.client_private_key")
              ? true
              : "Provide either a client secret or a client private key",
        }}
      >
        {(field) => (
          <Input.Password
            {...bindControl<string | undefined>(field)}
            placeholder={`Enter OAuth client secret${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Client Private Key (PEM)"
            tooltip="PEM private key signing the RFC 7523 private_key_jwt client assertion. Okta Cross App Access normally requires this. When set it takes precedence over the client secret."
          />
        }
        name="credentials.client_private_key"
      >
        {(field) => (
          <Input.TextArea
            {...bindControl<string | undefined>(field)}
            rows={3}
            placeholder={`-----BEGIN PRIVATE KEY-----${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Private Key ID (optional)"
            tooltip="The kid advertised in the client assertion JWT header, so the IdP can select the right registered key."
          />
        }
        name="credentials.client_private_key_id"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="my-signing-key-1"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Client Assertion Signing Algorithm (optional)"
            tooltip="Algorithm signing the client assertion JWT. Defaults to RS256."
          />
        }
        name="credentials.client_assertion_signing_alg"
      >
        {(field) => (
          <Input {...bindControl<string | undefined>(field)} placeholder="RS256" className={fieldClassName} />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Audience (optional)"
            tooltip="RFC 8693 audience sent on leg 1, identifying the upstream the ID-JAG assertion is minted for."
          />
        }
        name="audience"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="https://upstream.example.com"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Resource Indicator (optional)"
            tooltip="RFC 8707 resource indicator sent on leg 1. Separate from Audience, which is the RFC 8693 parameter."
          />
        }
        name="credentials.id_jag_resource"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="https://upstream.example.com/mcp"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Subject Token Type (optional)"
            tooltip="Type of the identity assertion exchanged on leg 1. Defaults to urn:ietf:params:oauth:token-type:id_token."
          />
        }
        name="subject_token_type"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="urn:ietf:params:oauth:token-type:id_token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={<FieldLabel label="Scopes (optional)" tooltip="Scopes requested on leg 1 of the exchange." />}
        name="credentials.scopes"
      >
        {(field) => (
          <Select
            {...bindControl<string[] | undefined>(field)}
            mode="tags"
            tokenSeparators={[","]}
            placeholder="Add scopes"
            className="rounded-lg"
            size="large"
          />
        )}
      </MountedFormField>
    </>
  );
};

export default IdJagFormFields;
