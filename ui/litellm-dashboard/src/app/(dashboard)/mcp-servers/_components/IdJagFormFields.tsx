import { useLanguage } from "@/contexts/LanguageContext";
import { Info } from "lucide-react";
import React from "react";
import { SimpleTooltip } from "@/components/ui/tooltip";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import UpstreamTokenHeaderField from "./UpstreamTokenHeaderField";
import { requiredRule } from "@/components/common_components/formRules";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { requiredUnlessSiblingSet, tagsControl, textControl } from "./mcpFieldRules";

interface IdJagFormFieldsProps {
  isEditing?: boolean;
}

const fieldClassName = "rounded-lg border-border focus:border-info focus:ring-ring";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-foreground flex items-center">
    {label}
    <SimpleTooltip content={tooltip}>
      <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
    </SimpleTooltip>
  </span>
);

const PRIVATE_KEY_PATH = ["credentials", "client_private_key"] as const;

const IdJagFormFields: React.FC<IdJagFormFieldsProps> = ({ isEditing = false }) => {
  const { t } = useLanguage();
  const placeholderSuffix = isEditing ? t(" (leave blank to keep existing)") : "";
  const requiredWhenCreating = (message: string) =>
    isEditing ? undefined : { validate: { required: requiredRule(t(message)) } };

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Org Token Endpoint (leg 1)")}
            tooltip={t(
              "Your IdP org authorization server's token endpoint. LiteLLM exchanges the user's identity assertion here for an ID-JAG assertion (RFC 8693 with requested_token_type=urn:ietf:params:oauth:token-type:id-jag).",
            )}
          />
        }
        name="token_exchange_endpoint"
        required={!isEditing}
        rules={requiredWhenCreating("The org token endpoint is required for ID-JAG")}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://your-org.okta.com/oauth2/v1/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Resource Token Endpoint (leg 2)")}
            tooltip={t(
              "The upstream resource authorization server's token endpoint. LiteLLM posts the ID-JAG assertion here as an RFC 7523 jwt-bearer grant to get the access token the MCP server accepts.",
            )}
          />
        }
        name={["credentials", "id_jag_resource_token_endpoint"]}
        required={!isEditing}
        rules={requiredWhenCreating("The resource token endpoint is required for ID-JAG")}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://upstream.example.com/oauth2/token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel label={t("Client ID")} tooltip={t("OAuth2 client ID LiteLLM authenticates as on both legs.")} />
        }
        name={["credentials", "client_id"]}
        required={!isEditing}
        rules={requiredWhenCreating("Client ID is required for ID-JAG")}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={t("Enter OAuth client ID{placeholderSuffix}", {
              placeholderSuffix: String(placeholderSuffix),
            })}
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Client Secret")}
            tooltip={t(
              "Authenticates LiteLLM as the OAuth client via client_secret_post. Leave blank when using a private key instead; a private key takes precedence over this secret.",
            )}
          />
        }
        name={["credentials", "client_secret"]}
        rules={
          isEditing
            ? undefined
            : {
                deps: ["credentials.client_private_key"],
                validate: {
                  secretOrPrivateKey: requiredUnlessSiblingSet(
                    PRIVATE_KEY_PATH,
                    t("Provide either a client secret or a client private key"),
                  ),
                },
              }
        }
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={t("Enter OAuth client secret{placeholderSuffix}", {
              placeholderSuffix: String(placeholderSuffix),
            })}
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Client Private Key (PEM)")}
            tooltip={t(
              "PEM private key signing the RFC 7523 private_key_jwt client assertion. Okta Cross App Access normally requires this. When set it takes precedence over the client secret.",
            )}
          />
        }
        name={PRIVATE_KEY_PATH}
      >
        {(control) => (
          <Textarea
            {...textControl(control)}
            rows={3}
            placeholder={`-----BEGIN PRIVATE KEY-----${placeholderSuffix}`}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Private Key ID (optional)")}
            tooltip={t(
              "The kid advertised in the client assertion JWT header, so the IdP can select the right registered key.",
            )}
          />
        }
        name={["credentials", "client_private_key_id"]}
      >
        {(control) => <Input {...textControl(control)} placeholder="my-signing-key-1" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Client Assertion Signing Algorithm (optional)")}
            tooltip={t("Algorithm signing the client assertion JWT. Defaults to RS256.")}
          />
        }
        name={["credentials", "client_assertion_signing_alg"]}
      >
        {(control) => <Input {...textControl(control)} placeholder="RS256" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Audience (optional)")}
            tooltip={t("RFC 8693 audience sent on leg 1, identifying the upstream the ID-JAG assertion is minted for.")}
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
            label={t("Resource Indicator (optional)")}
            tooltip={t(
              "RFC 8707 resource indicator sent on leg 1. Separate from Audience, which is the RFC 8693 parameter.",
            )}
          />
        }
        name={["credentials", "id_jag_resource"]}
      >
        {(control) => (
          <Input {...textControl(control)} placeholder="https://upstream.example.com/mcp" className={fieldClassName} />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("Subject Token Type (optional)")}
            tooltip={t(
              "Type of the identity assertion exchanged on leg 1. Defaults to urn:ietf:params:oauth:token-type:id_token.",
            )}
          />
        }
        name="subject_token_type"
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="urn:ietf:params:oauth:token-type:id_token"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={<FieldLabel label={t("Scopes (optional)")} tooltip={t("Scopes requested on leg 1 of the exchange.")} />}
        name={["credentials", "scopes"]}
      >
        {(control) => <MultiSelect {...tagsControl(control, t)} placeholder={t("Add scopes")} className="rounded-lg" />}
      </MountedFormField>
      <UpstreamTokenHeaderField />
    </>
  );
};

export default IdJagFormFields;
