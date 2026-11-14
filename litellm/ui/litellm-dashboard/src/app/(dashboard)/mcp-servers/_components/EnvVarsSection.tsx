import type { ParseKeys } from "i18next";
import { CircleMinus, Info, Plus } from "lucide-react";
import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { useFieldArray, useFormContext, useWatch } from "react-hook-form";

import {
  MountedFormField,
  useMountedName,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { requiredRule } from "@/components/common_components/formRules";
import { matchesPattern, selectControl, selectTriggerControl, textControl } from "./mcpFieldRules";
import { listControl } from "./mcpFormStore";

const SCOPE_OPTIONS: ReadonlyArray<{ value: string; labelKey: ParseKeys }> = [
  { value: "global", labelKey: "mcpServers.envVarsSection.scopes.instance" },
  { value: "user", labelKey: "mcpServers.envVarsSection.scopes.perUser" },
];

const VARIABLE_SYNTAX = "${VAR_NAME}";

const VARIABLE_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

/**
 * Form section for admin-configured MCP environment variables.
 *
 * Each row has: name | value | scope. Variables can be interpolated into
 * Static Headers via ${NAME}. ``scope=global`` (shown as "Instance") values
 * are used as-is. ``scope=user`` (shown as "Per-user") values are filled in
 * by each user via the MCP Gateway dashboard.
 *
 * The parent form reads the ``env_vars`` field from the form values.
 */
const EnvVarsSection: React.FC = () => {
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const { fields, append, remove } = useFieldArray({ control: listControl(control), name: "env_vars" });
  useMountedName("env_vars");
  const scopeOptions = SCOPE_OPTIONS.map((option) => ({ value: option.value, label: t(option.labelKey) }));

  return (
    <div className="rounded-lg border border-border bg-muted p-4">
      <div className="flex items-center gap-2 mb-1">
        <strong className="text-sm font-semibold">{t("mcpServers.envVarsSection.title")}</strong>
        <SimpleTooltip
          content={
            <Trans
              i18nKey="mcpServers.envVarsSection.tooltip"
              values={{ variableSyntax: VARIABLE_SYNTAX }}
              components={{ code: <code />, b: <b /> }}
            />
          }
        >
          <Info className="size-4 text-info hover:text-info/80 cursor-help" />
        </SimpleTooltip>
      </div>
      <span className="mb-3 block text-xs text-muted-foreground">
        <Trans
          i18nKey="mcpServers.envVarsSection.reference"
          values={{ variableSyntax: VARIABLE_SYNTAX }}
          components={{ code: <code /> }}
        />{" "}
        <code className="bg-card px-1 rounded-sm border border-border">
          {"${DB_PROTOCOL}://${CORP_USERNAME}:${CORP_PASSWORD}@${DB_HOSTNAME}"}
        </code>
      </span>

      <div className="space-y-2">
        {fields.length > 0 && (
          <div className="flex gap-3 px-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
            <div style={{ flex: 1 }}>{t("mcpServers.envVarsSection.columns.name")}</div>
            <div style={{ flex: 1 }}>{t("mcpServers.envVarsSection.columns.value")}</div>
            <div style={{ width: 160 }}>{t("mcpServers.envVarsSection.columns.scope")}</div>
            <div style={{ width: 24 }} />
          </div>
        )}
        {fields.map((item, index) => (
          <div key={item.id} className="flex gap-3 items-start">
            <MountedFormField
              name={["env_vars", String(index), "name"]}
              className="mb-0 flex-1"
              rules={{
                validate: {
                  required: requiredRule(t("mcpServers.envVarsSection.name.required")),
                  pattern: matchesPattern(VARIABLE_NAME_PATTERN, t("mcpServers.envVarsSection.name.invalid")),
                },
              }}
            >
              {(control) => (
                <Input
                  {...textControl(control)}
                  placeholder={t("mcpServers.envVarsSection.name.placeholder")}
                  className="rounded-md font-mono"
                />
              )}
            </MountedFormField>
            <div style={{ flex: 1 }}>
              <ScopedValueOrDescription index={index} />
            </div>
            <MountedFormField name={["env_vars", String(index), "scope"]} className="mb-0 w-40" defaultValue="global">
              {(control) => (
                <Select {...selectControl<string>(control)} items={scopeOptions}>
                  <SelectTrigger {...selectTriggerControl(control)} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {scopeOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </MountedFormField>
            <div style={{ width: 24, height: 32 }} className="flex items-center justify-center">
              <CircleMinus
                onClick={() => remove(index)}
                className="size-4 text-muted-foreground hover:text-destructive cursor-pointer"
              />
            </div>
          </div>
        ))}
        <Button variant="outline" className="w-full border-dashed" onClick={() => append({ scope: "global" })}>
          <Plus />
          {t("mcpServers.envVarsSection.addVariable")}
        </Button>
      </div>
    </div>
  );
};

// For instance-scoped vars this column holds the admin value. For per-user
// vars the value comes from each user later, so the column instead captures an
// optional description that the per-user fill-in modal shows as a hint.
const ScopedValueOrDescription: React.FC<{ index: number }> = ({ index }) => {
  const { t } = useTranslation();
  const isPerUser = useWatch({ name: `env_vars.${index}.scope` }) === "user";
  if (isPerUser) {
    return (
      <MountedFormField name={["env_vars", String(index), "description"]} className="mb-0">
        {(control) => (
          <InputGroup>
            <InputGroupAddon>
              <SimpleTooltip content={t("mcpServers.envVarsSection.hint.tooltip")}>
                <span className="text-xs text-muted-foreground cursor-help whitespace-nowrap">
                  <Info className="mr-1 inline size-3 align-text-bottom" />
                  {t("mcpServers.envVarsSection.hint.label")}
                </span>
              </SimpleTooltip>
            </InputGroupAddon>
            <InputGroupInput
              {...textControl(control)}
              placeholder={t("mcpServers.envVarsSection.hint.placeholder")}
              className="text-muted-foreground"
            />
          </InputGroup>
        )}
      </MountedFormField>
    );
  }
  return (
    <MountedFormField name={["env_vars", String(index), "value"]} className="mb-0">
      {(control) => (
        <Input
          {...textControl(control)}
          placeholder={t("mcpServers.envVarsSection.value.placeholder")}
          className="rounded-md font-mono"
        />
      )}
    </MountedFormField>
  );
};

export default EnvVarsSection;
