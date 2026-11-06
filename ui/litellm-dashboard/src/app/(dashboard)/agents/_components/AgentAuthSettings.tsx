import React from "react";
import { useFieldArray, useFormContext, useWatch } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { ChevronDown, Plus, Trash2 } from "lucide-react";
import { AUTH_TYPE } from "@/components/mcp_tools/types";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Field, FieldGroup, FieldTitle } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AgentFormField, AgentFormValues, AgentTagsInput, labelWithHint } from "./AgentFormKit";
import { AGENT_AUTH_TYPE_ITEMS, AgentAuthSupport, isAuthValueMissing } from "./agent_auth";

const StaticHeadersFieldArray = () => {
  const { t } = useTranslation();
  const { control } = useFormContext<AgentFormValues>();
  const { fields, append, remove } = useFieldArray({ control, name: "static_headers" });

  return (
    <>
      {fields.map((item, index) => (
        <div key={item.id} className="flex items-start gap-2">
          <AgentFormField
            name={`static_headers.${index}.header`}
            rules={{ required: t("agents.formFields.staticHeaders.errors.nameRequired") }}
          >
            {({ value, onChange, ref, ...control }) => (
              <Input
                {...control}
                ref={ref}
                className="w-55"
                placeholder={t("agents.formFields.staticHeaders.namePlaceholder")}
                value={typeof value === "string" ? value : ""}
                onChange={onChange}
              />
            )}
          </AgentFormField>
          <AgentFormField
            name={`static_headers.${index}.value`}
            rules={{ required: t("agents.formFields.staticHeaders.errors.valueRequired") }}
          >
            {({ value, onChange, ref, ...control }) => (
              <Input
                {...control}
                ref={ref}
                className="w-65"
                placeholder={t("agents.formFields.staticHeaders.valuePlaceholder")}
                value={typeof value === "string" ? value : ""}
                onChange={onChange}
              />
            )}
          </AgentFormField>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={t("agents.formFields.staticHeaders.remove")}
            className="text-destructive hover:text-destructive/80"
            onClick={() => remove(index)}
          >
            <Trash2 />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" className="w-full border-dashed" onClick={() => append({})}>
        <Plus />
        {t("agents.formFields.staticHeaders.add")}
      </Button>
    </>
  );
};

const AgentAuthSettings = ({ support }: { support: AgentAuthSupport }) => {
  const { t } = useTranslation();
  const { control } = useFormContext<AgentFormValues>();
  const authType = useWatch({ control, name: "auth_type" });
  const agentName = useWatch({ control, name: "agent_name" });

  if (!support.bearer && !support.headers) return null;

  const authTypeHint =
    authType === AUTH_TYPE.BEARER_TOKEN
      ? t("agents.formFields.authHeaders.authTypeBearerHint")
      : t("agents.formFields.authHeaders.authTypeNoneHint", {
          agentName: typeof agentName === "string" && agentName ? agentName : "<agent-name>",
        });

  return (
    <Collapsible defaultOpen className="mb-4 rounded-md border border-border px-4">
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-4 py-3 text-left">
        <span className="text-sm font-semibold text-foreground">{t("agents.formFields.authHeaders.title")}</span>
        <ChevronDown className="size-4 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent keepMounted>
        <FieldGroup className="pb-5">
          {support.bearer && (
            <AgentFormField
              name="auth_type"
              label={t("agents.formFields.authHeaders.authType")}
              defaultValue={AUTH_TYPE.NONE}
              description={authTypeHint}
            >
              {({ value, onChange, ...control }) => (
                <Select
                  items={AGENT_AUTH_TYPE_ITEMS}
                  value={typeof value === "string" ? value : AUTH_TYPE.NONE}
                  onValueChange={(next) => next !== null && onChange(next)}
                >
                  <SelectTrigger id={control.id} aria-invalid={control["aria-invalid"]} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {AGENT_AUTH_TYPE_ITEMS.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </AgentFormField>
          )}

          {support.bearer && authType === AUTH_TYPE.BEARER_TOKEN && (
            <AgentFormField
              name="api_key"
              label={labelWithHint(
                t("agents.formFields.authHeaders.authValue"),
                t("agents.formFields.authHeaders.authValueHint"),
              )}
              rules={{
                validate: (value, values) =>
                  isAuthValueMissing(value, values) ? t("agents.formFields.authHeaders.authValueRequired") : true,
              }}
            >
              {({ value, onChange, ref, ...control }) => (
                <PasswordInput
                  {...control}
                  ref={ref}
                  value={typeof value === "string" ? value : ""}
                  onChange={onChange}
                  placeholder={t("agents.formFields.authHeaders.authValuePlaceholder")}
                />
              )}
            </AgentFormField>
          )}

          {support.headers && (
            <>
              <Field>
                <FieldTitle>
                  {labelWithHint(
                    t("agents.formFields.authHeaders.staticHeaders"),
                    t("agents.formFields.authHeaders.staticHeadersHint"),
                  )}
                </FieldTitle>
                <div className="flex flex-col gap-2">
                  <StaticHeadersFieldArray />
                </div>
              </Field>

              <AgentFormField
                name="extra_headers"
                label={labelWithHint(
                  t("agents.formFields.authHeaders.forwardHeaders"),
                  t("agents.formFields.authHeaders.forwardHeadersHint"),
                )}
              >
                {({ id, value, onChange }) => (
                  <AgentTagsInput
                    id={id}
                    value={Array.isArray(value) ? (value as string[]) : []}
                    onValueChange={onChange}
                    placeholder={t("agents.formFields.authHeaders.forwardHeadersPlaceholder")}
                  />
                )}
              </AgentFormField>
            </>
          )}
        </FieldGroup>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default AgentAuthSettings;
