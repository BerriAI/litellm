import { Info } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
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
const PEM_HEADER = "-----BEGIN PRIVATE KEY-----";

const IdJagFormFields: React.FC<IdJagFormFieldsProps> = ({ isEditing = false }) => {
  const { t } = useTranslation();
  const requiredWhenCreating = (message: string) =>
    isEditing ? undefined : { validate: { required: requiredRule(message) } };

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.orgTokenEndpoint.label")}
            tooltip={t("mcpServers.idJagFormFields.orgTokenEndpoint.tooltip")}
          />
        }
        name="token_exchange_endpoint"
        required={!isEditing}
        rules={requiredWhenCreating(t("mcpServers.idJagFormFields.orgTokenEndpoint.required"))}
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
            label={t("mcpServers.idJagFormFields.resourceTokenEndpoint.label")}
            tooltip={t("mcpServers.idJagFormFields.resourceTokenEndpoint.tooltip")}
          />
        }
        name={["credentials", "id_jag_resource_token_endpoint"]}
        required={!isEditing}
        rules={requiredWhenCreating(t("mcpServers.idJagFormFields.resourceTokenEndpoint.required"))}
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
          <FieldLabel
            label={t("mcpServers.idJagFormFields.clientId.label")}
            tooltip={t("mcpServers.idJagFormFields.clientId.tooltip")}
          />
        }
        name={["credentials", "client_id"]}
        required={!isEditing}
        rules={requiredWhenCreating(t("mcpServers.idJagFormFields.clientId.required"))}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={
              isEditing
                ? t("mcpServers.idJagFormFields.clientId.placeholderEditing")
                : t("mcpServers.idJagFormFields.clientId.placeholder")
            }
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.clientSecret.label")}
            tooltip={t("mcpServers.idJagFormFields.clientSecret.tooltip")}
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
                    t("mcpServers.idJagFormFields.clientSecret.secretOrPrivateKey"),
                  ),
                },
              }
        }
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={
              isEditing
                ? t("mcpServers.idJagFormFields.clientSecret.placeholderEditing")
                : t("mcpServers.idJagFormFields.clientSecret.placeholder")
            }
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.privateKey.label")}
            tooltip={t("mcpServers.idJagFormFields.privateKey.tooltip")}
          />
        }
        name={PRIVATE_KEY_PATH}
      >
        {(control) => (
          <Textarea
            {...textControl(control)}
            rows={3}
            placeholder={
              isEditing
                ? t("mcpServers.idJagFormFields.privateKey.placeholderEditing", { pemHeader: PEM_HEADER })
                : PEM_HEADER
            }
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.privateKeyId.label")}
            tooltip={t("mcpServers.idJagFormFields.privateKeyId.tooltip")}
          />
        }
        name={["credentials", "client_private_key_id"]}
      >
        {(control) => <Input {...textControl(control)} placeholder="my-signing-key-1" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.signingAlg.label")}
            tooltip={t("mcpServers.idJagFormFields.signingAlg.tooltip")}
          />
        }
        name={["credentials", "client_assertion_signing_alg"]}
      >
        {(control) => <Input {...textControl(control)} placeholder="RS256" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.audience.label")}
            tooltip={t("mcpServers.idJagFormFields.audience.tooltip")}
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
            label={t("mcpServers.idJagFormFields.resource.label")}
            tooltip={t("mcpServers.idJagFormFields.resource.tooltip")}
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
            label={t("mcpServers.idJagFormFields.subjectTokenType.label")}
            tooltip={t("mcpServers.idJagFormFields.subjectTokenType.tooltip")}
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
        label={
          <FieldLabel
            label={t("mcpServers.idJagFormFields.scopes.label")}
            tooltip={t("mcpServers.idJagFormFields.scopes.tooltip")}
          />
        }
        name={["credentials", "scopes"]}
      >
        {(control) => (
          <MultiSelect
            {...tagsControl(control, t)}
            placeholder={t("mcpServers.idJagFormFields.scopes.placeholder")}
            className="rounded-lg"
          />
        )}
      </MountedFormField>
      <UpstreamTokenHeaderField />
    </>
  );
};

export default IdJagFormFields;
