import React from "react";
import { useFieldArray, useFormContext, useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Field, FieldTitle } from "@/components/ui/field";
import { PasswordInput } from "@/components/shared/PasswordInput";
import i18n from "@/i18n/i18n";
import { AgentFormField, AgentFormValues, labelWithHint } from "./AgentFormKit";
import { KILL_SWITCH_AUTH_TYPES, KILL_SWITCH_METHODS, validateKillSwitchBody } from "./kill_switch_config";

const KeyValueFieldArray = ({
  name,
  addLabel,
  keyPlaceholder,
  valuePlaceholder,
}: {
  name: "kill_switch.headers" | "kill_switch.query_params";
  addLabel: string;
  keyPlaceholder: string;
  valuePlaceholder: string;
}) => {
  const { t } = useTranslation();
  const { control } = useFormContext<AgentFormValues>();
  const { fields, append, remove } = useFieldArray({ control, name });

  return (
    <div className="flex flex-col gap-2">
      {fields.map((item, index) => (
        <div key={item.id} className="flex items-start gap-2">
          <AgentFormField
            name={`${name}.${index}.key`}
            rules={{ required: t("agents.killSwitchFields.errors.nameRequired") }}
          >
            {({ value, onChange, ref, ...control }) => (
              <Input
                {...control}
                ref={ref}
                className="w-55"
                placeholder={keyPlaceholder}
                value={typeof value === "string" ? value : ""}
                onChange={onChange}
              />
            )}
          </AgentFormField>
          <AgentFormField name={`${name}.${index}.value`}>
            {({ value, onChange, ref, ...control }) => (
              <Input
                {...control}
                ref={ref}
                className="w-65"
                placeholder={valuePlaceholder}
                value={typeof value === "string" ? value : ""}
                onChange={onChange}
              />
            )}
          </AgentFormField>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t("agents.killSwitchFields.removeItem", { item: addLabel.replace(/^Add /, "").toLowerCase() })}
            className="text-destructive hover:text-destructive/80"
            onClick={() => remove(index)}
          >
            <Trash2 />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" className="w-full border-dashed" onClick={() => append({})}>
        <Plus />
        {addLabel}
      </Button>
    </div>
  );
};

const TextField = ({
  name,
  label,
  placeholder,
  required,
  secret,
}: {
  name: `kill_switch.${string}`;
  label: React.ReactNode;
  placeholder?: string;
  required?: string;
  secret?: boolean;
}) => (
  <AgentFormField name={name} label={label} rules={required ? { required } : undefined}>
    {({ value, onChange, ref, ...control }) =>
      secret ? (
        <PasswordInput
          {...control}
          ref={ref}
          placeholder={placeholder}
          value={typeof value === "string" ? value : ""}
          onChange={onChange}
        />
      ) : (
        <Input
          {...control}
          ref={ref}
          placeholder={placeholder}
          value={typeof value === "string" ? value : ""}
          onChange={onChange}
        />
      )
    }
  </AgentFormField>
);

const KillSwitchAuthFields = () => {
  const { t } = useTranslation();
  const { control } = useFormContext<AgentFormValues>();
  const authType = useWatch({ control, name: "kill_switch.auth_type" });

  switch (authType) {
    case "bearer":
      return (
        <TextField
          name="kill_switch.auth_token"
          label={t("agents.killSwitchFields.auth.bearerToken")}
          required={t("agents.killSwitchFields.errors.tokenRequired")}
          secret
        />
      );
    case "api_key":
      return (
        <>
          <TextField
            name="kill_switch.auth_header_name"
            label={t("agents.killSwitchFields.auth.headerName")}
            placeholder="X-API-Key"
          />
          <TextField
            name="kill_switch.auth_api_key"
            label={t("agents.killSwitchFields.auth.apiKey")}
            required={t("agents.killSwitchFields.errors.apiKeyRequired")}
            secret
          />
        </>
      );
    case "basic":
      return (
        <>
          <TextField
            name="kill_switch.auth_username"
            label={t("agents.killSwitchFields.auth.username")}
            required={t("agents.killSwitchFields.errors.usernameRequired")}
          />
          <TextField
            name="kill_switch.auth_password"
            label={t("agents.killSwitchFields.auth.password")}
            required={t("agents.killSwitchFields.errors.passwordRequired")}
            secret
          />
        </>
      );
    default:
      return null;
  }
};

const KillSwitchFormFields = () => (
  <>
    <TextField
      name="kill_switch.url"
      label={labelWithHint(
        i18n.t("agents.killSwitchFields.webhookUrl.label"),
        i18n.t("agents.killSwitchFields.webhookUrl.hint"),
      )}
      placeholder="https://example.com/hooks/kill-agent"
    />

    <AgentFormField name="kill_switch.method" label={i18n.t("agents.killSwitchFields.method")}>
      {({ value, onChange, ref: _ref, ...control }) => (
        <Select value={typeof value === "string" ? value : "POST"} onValueChange={onChange}>
          <SelectTrigger {...control} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {KILL_SWITCH_METHODS.map((method) => (
              <SelectItem key={method} value={method}>
                {method}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </AgentFormField>

    <Field>
      <FieldTitle>{i18n.t("agents.killSwitchFields.headers.title")}</FieldTitle>
      <KeyValueFieldArray
        name="kill_switch.headers"
        addLabel={i18n.t("agents.killSwitchFields.headers.add")}
        keyPlaceholder={i18n.t("agents.killSwitchFields.headers.keyPlaceholder")}
        valuePlaceholder={i18n.t("agents.killSwitchFields.headers.valuePlaceholder")}
      />
    </Field>

    <Field>
      <FieldTitle>{i18n.t("agents.killSwitchFields.queryParams.title")}</FieldTitle>
      <KeyValueFieldArray
        name="kill_switch.query_params"
        addLabel={i18n.t("agents.killSwitchFields.queryParams.add")}
        keyPlaceholder={i18n.t("agents.killSwitchFields.queryParams.keyPlaceholder")}
        valuePlaceholder={i18n.t("agents.killSwitchFields.queryParams.valuePlaceholder")}
      />
    </Field>

    <AgentFormField
      name="kill_switch.body"
      label={labelWithHint(i18n.t("agents.killSwitchFields.body.label"), i18n.t("agents.killSwitchFields.body.hint"))}
      rules={{ validate: (value) => validateKillSwitchBody(typeof value === "string" ? value : "") }}
    >
      {({ value, onChange, ref, ...control }) => (
        <Textarea
          {...control}
          ref={ref}
          rows={4}
          placeholder='{"reason": "manual kill switch"}'
          value={typeof value === "string" ? value : ""}
          onChange={onChange}
        />
      )}
    </AgentFormField>

    <AgentFormField name="kill_switch.auth_type" label={i18n.t("agents.killSwitchFields.authentication")}>
      {({ value, onChange, ref: _ref, ...control }) => (
        <Select value={typeof value === "string" ? value : "none"} onValueChange={onChange}>
          <SelectTrigger {...control} className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {KILL_SWITCH_AUTH_TYPES.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {i18n.t(option.labelKey)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </AgentFormField>

    <KillSwitchAuthFields />
  </>
);

export default KillSwitchFormFields;
