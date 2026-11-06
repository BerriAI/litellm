import { Info } from "lucide-react";
import React from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import UpstreamTokenHeaderField from "./UpstreamTokenHeaderField";
import { requiredRule } from "@/components/common_components/formRules";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { selectControl, selectTriggerControl, tagsControl, textControl } from "./mcpFieldRules";

interface TokenExchangeFormFieldsProps {
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

const TokenExchangeFormFields: React.FC<TokenExchangeFormFieldsProps> = ({ isEditing = false }) => {
  const { t } = useTranslation();
  const tokenExchangeProfileItems = [
    { value: "rfc8693", label: t("mcpServers.tokenExchangeFormFields.profile.rfc8693") },
    { value: "entra_obo", label: "Microsoft Entra OBO" },
  ];
  const isEntraObo = useWatch({ name: "token_exchange_profile" }) === "entra_obo";
  const requiredWhenCreating = (message: string) =>
    isEditing ? undefined : { validate: { required: requiredRule(message) } };

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.tokenExchangeFormFields.profile.label")}
            tooltip={t("mcpServers.tokenExchangeFormFields.profile.tooltip")}
          />
        }
        name="token_exchange_profile"
        {...(isEditing ? {} : { defaultValue: "rfc8693" })}
      >
        {(control) => (
          <Select {...selectControl<string>(control)} items={tokenExchangeProfileItems}>
            <SelectTrigger {...selectTriggerControl(control)} className="w-full rounded-lg">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {tokenExchangeProfileItems.map((item) => (
                <SelectItem key={item.value} value={item.value}>
                  <span className="font-medium">{item.label}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.tokenExchangeFormFields.endpoint.label")}
            tooltip={t("mcpServers.tokenExchangeFormFields.endpoint.tooltip")}
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
            label={t("mcpServers.tokenExchangeFormFields.clientId.label")}
            tooltip={t("mcpServers.tokenExchangeFormFields.clientId.tooltip")}
          />
        }
        name={["credentials", "client_id"]}
        required={!isEditing}
        rules={requiredWhenCreating(t("mcpServers.tokenExchangeFormFields.clientId.required"))}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={
              isEditing
                ? t("mcpServers.tokenExchangeFormFields.clientId.placeholderEditing")
                : t("mcpServers.tokenExchangeFormFields.clientId.placeholder")
            }
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.tokenExchangeFormFields.clientSecret.label")}
            tooltip={t("mcpServers.tokenExchangeFormFields.clientSecret.tooltip")}
          />
        }
        name={["credentials", "client_secret"]}
        required={!isEditing}
        rules={requiredWhenCreating(t("mcpServers.tokenExchangeFormFields.clientSecret.required"))}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={
              isEditing
                ? t("mcpServers.tokenExchangeFormFields.clientSecret.placeholderEditing")
                : t("mcpServers.tokenExchangeFormFields.clientSecret.placeholder")
            }
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      {!isEntraObo && (
        <>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.tokenExchangeFormFields.audience.label")}
                tooltip={t("mcpServers.tokenExchangeFormFields.audience.tooltip")}
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
                label={t("mcpServers.tokenExchangeFormFields.subjectTokenType.label")}
                tooltip={t("mcpServers.tokenExchangeFormFields.subjectTokenType.tooltip")}
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
            label={
              isEntraObo
                ? t("mcpServers.tokenExchangeFormFields.scopes.labelEntraObo")
                : t("mcpServers.tokenExchangeFormFields.scopes.label")
            }
            tooltip={
              isEntraObo
                ? t("mcpServers.tokenExchangeFormFields.scopes.tooltipEntraObo")
                : t("mcpServers.tokenExchangeFormFields.scopes.tooltip")
            }
          />
        }
        name={["credentials", "scopes"]}
        required={isEntraObo}
        rules={
          isEntraObo
            ? {
                validate: {
                  required: requiredRule(t("mcpServers.tokenExchangeFormFields.scopes.requiredEntraObo")),
                },
              }
            : undefined
        }
      >
        {(control) => (
          <MultiSelect
            {...tagsControl(control, t)}
            placeholder={
              isEntraObo ? "api://<app-id>/.default" : t("mcpServers.tokenExchangeFormFields.scopes.placeholder")
            }
            className="rounded-lg"
          />
        )}
      </MountedFormField>
      <UpstreamTokenHeaderField />
    </>
  );
};

export default TokenExchangeFormFields;
