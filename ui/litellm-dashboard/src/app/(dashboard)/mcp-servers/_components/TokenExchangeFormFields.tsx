import React from "react";
import { Input, Select, Tooltip } from "antd";
import { Info } from "lucide-react";
import { useWatch } from "react-hook-form";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { antdRequired } from "@/components/common_components/antdFormRules";
import { selectControl, textControl } from "./mcpFieldRules";

interface TokenExchangeFormFieldsProps {
  isEditing?: boolean;
}

const fieldClassName = "rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-gray-700 flex items-center">
    {label}
    <Tooltip title={tooltip}>
      <Info className="ml-2 size-4 text-blue-400 hover:text-blue-600 cursor-help" />
    </Tooltip>
  </span>
);

const TokenExchangeFormFields: React.FC<TokenExchangeFormFieldsProps> = ({ isEditing = false }) => {
  const placeholderSuffix = isEditing ? " (leave blank to keep existing)" : "";
  const isEntraObo = useWatch({ name: "token_exchange_profile" }) === "entra_obo";
  const requiredWhenCreating = (message: string) =>
    isEditing ? undefined : { validate: { required: antdRequired(message) } };

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label="Profile"
            tooltip="Token-exchange wire dialect. RFC 8693 is the standard token-exchange grant. Microsoft Entra OBO uses Entra's On-Behalf-Of dialect (the RFC 7523 jwt-bearer grant with requested_token_use=on_behalf_of) and carries the target resource in a scope like api://<app-id>/.default."
          />
        }
        name="token_exchange_profile"
        {...(isEditing ? {} : { defaultValue: "rfc8693" })}
      >
        {(control) => (
          <Select {...selectControl(control)} className="rounded-lg" size="large">
            <Select.Option value="rfc8693">
              <span className="font-medium">RFC 8693 (standard)</span>
            </Select.Option>
            <Select.Option value="entra_obo">
              <span className="font-medium">Microsoft Entra OBO</span>
            </Select.Option>
          </Select>
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Token Exchange Endpoint (optional)"
            tooltip="RFC 8693 token endpoint. The proxy exchanges the user's incoming token here for a scoped token used to call the upstream MCP server. Leave blank to auto-discover it from the upstream's protected-resource metadata (RFC 9728 then RFC 8414)."
          />
        }
        name="token_exchange_endpoint"
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://idp.example.com/oauth2/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Client ID"
            tooltip="OAuth2 client ID used to authenticate to the token exchange endpoint."
          />
        }
        name={["credentials", "client_id"]}
        required={!isEditing}
        rules={requiredWhenCreating("Client ID is required for token exchange")}
      >
        {(control) => (
          <Input.Password
            {...textControl(control)}
            placeholder={`Enter OAuth client ID${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="Client Secret"
            tooltip="OAuth2 client secret used to authenticate to the token exchange endpoint."
          />
        }
        name={["credentials", "client_secret"]}
        required={!isEditing}
        rules={requiredWhenCreating("Client Secret is required for token exchange")}
      >
        {(control) => (
          <Input.Password
            {...textControl(control)}
            placeholder={`Enter OAuth client secret${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      {!isEntraObo && (
        <>
          <MountedFormField
            label={
              <FieldLabel
                label="Audience (optional)"
                tooltip="Target audience for the exchanged token (RFC 8693 audience). Identifies the upstream MCP server the token is for."
              />
            }
            name="audience"
          >
            {(control) => (
              <Input {...textControl(control)} placeholder="https://upstream.example.com" className={fieldClassName} />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label="Subject Token Type (optional)"
                tooltip="Type of the user's incoming token (RFC 8693 subject_token_type). Defaults to urn:ietf:params:oauth:token-type:access_token."
              />
            }
            name="subject_token_type"
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder="urn:ietf:params:oauth:token-type:access_token"
                className={fieldClassName}
              />
            )}
          </MountedFormField>
        </>
      )}
      <MountedFormField
        label={
          <FieldLabel
            label={isEntraObo ? "Scopes" : "Scopes (optional)"}
            tooltip={
              isEntraObo
                ? "Microsoft Entra OBO carries the target resource in the scope, so at least one is required (e.g. api://<app-id>/.default)."
                : "Optional scopes to request during the token exchange."
            }
          />
        }
        name={["credentials", "scopes"]}
        required={isEntraObo}
        rules={
          isEntraObo
            ? {
                validate: {
                  required: antdRequired("Microsoft Entra OBO requires a scope, e.g. api://<app-id>/.default"),
                },
              }
            : undefined
        }
      >
        {(control) => (
          <Select
            {...selectControl(control)}
            mode="tags"
            tokenSeparators={[","]}
            placeholder={isEntraObo ? "api://<app-id>/.default" : "Add scopes"}
            className="rounded-lg"
            size="large"
          />
        )}
      </MountedFormField>
    </>
  );
};

export default TokenExchangeFormFields;
