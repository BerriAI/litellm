import React from "react";
import { useForm, type Control } from "react-hook-form";
import type { ParseKeys, TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { CircleHelp } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { FieldGroup } from "@/components/ui/field";
import { FormField, type FormFieldControlProps } from "@/components/shared/form/FormField";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { InputSchemaProperty } from "@/components/mcp_tools/types";
import {
  ToolArgumentField,
  ToolArgumentsFormValues,
  buildToolCallArguments,
  initialArgumentValues,
  resolveSchemaProperty,
  toolArgumentsResolver,
} from "./toolCallArguments";

const argumentLabel = (field: ToolArgumentField): React.ReactNode => (
  <span className="flex items-center">
    {field.key}
    {field.required && <span className="ml-1 text-destructive">*</span>}
    {field.prop.description && (
      <Tooltip>
        <TooltipTrigger render={<CircleHelp className="ml-2 size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
        <TooltipContent>{field.prop.description}</TooltipContent>
      </Tooltip>
    )}
  </span>
);

const BOOLEAN_ITEMS: readonly { readonly value: boolean; readonly labelKey: ParseKeys }[] = [
  { value: true, labelKey: "mcpServers.toolArgumentsForm.boolean.true" },
  { value: false, labelKey: "mcpServers.toolArgumentsForm.boolean.false" },
];

const booleanTitle = (value: unknown, t: TFunction): string | undefined => {
  if (value === true) return t("mcpServers.toolArgumentsForm.boolean.true");
  if (value === false) return t("mcpServers.toolArgumentsForm.boolean.false");
  return undefined;
};

const JsonArgumentControl: React.FC<{
  field: ToolArgumentField;
  prop: InputSchemaProperty;
  control: FormFieldControlProps<ToolArgumentsFormValues, `args.${number}`>;
}> = ({ field, prop, control }) => {
  const { t } = useTranslation();
  const isObject = prop.type === "object";
  const fallbackPlaceholder = isObject
    ? t("mcpServers.toolArgumentsForm.jsonObjectPlaceholder", { field: field.key })
    : t("mcpServers.toolArgumentsForm.jsonArrayPlaceholder", { field: field.key });
  return (
    <div className="space-y-2">
      <Textarea
        {...control}
        rows={isObject ? 6 : 4}
        value={(control.value as string) ?? ""}
        placeholder={prop.description || fallbackPlaceholder}
        spellCheck={false}
        data-testid={`textarea-${field.key}`}
        className="rounded-lg font-mono"
      />
      <p className="text-xs text-muted-foreground">
        {isObject ? t("mcpServers.toolArgumentsForm.jsonObjectHint") : t("mcpServers.toolArgumentsForm.jsonArrayHint")}
      </p>
    </div>
  );
};

const ToolArgumentControl: React.FC<{
  field: ToolArgumentField;
  control: FormFieldControlProps<ToolArgumentsFormValues, `args.${number}`>;
}> = ({ field, control }) => {
  const { t } = useTranslation();
  const prop = resolveSchemaProperty(field.prop);

  if (prop.type === "string" && prop.enum) {
    return (
      <select
        {...control}
        value={control.value == null ? -1 : prop.enum.indexOf(String(control.value))}
        onChange={(event) => control.onChange(prop.enum?.[Number(event.target.value)] ?? null)}
        className="w-full rounded-lg border border-input bg-transparent px-3 py-2 text-sm shadow-xs transition-colors focus:border-ring focus:ring-3 focus:ring-ring/50 focus:outline-hidden"
      >
        <option value={-1} disabled={field.required}>
          {t("mcpServers.toolArgumentsForm.selectPlaceholder", { field: field.key })}
        </option>
        {prop.enum.map((option, index) => (
          <option key={option} value={index}>
            {option === "" ? t("mcpServers.toolArgumentsForm.emptyString") : option}
          </option>
        ))}
      </select>
    );
  }

  if (prop.type === "number" || prop.type === "integer") {
    return (
      <Input
        {...control}
        type="number"
        step={prop.type === "integer" ? 1 : "any"}
        value={(control.value as number | string) ?? ""}
        placeholder={prop.description || t("mcpServers.toolArgumentsForm.enterPlaceholder", { field: field.key })}
        className="rounded-lg"
      />
    );
  }

  if (prop.type === "boolean") {
    const booleanItems = BOOLEAN_ITEMS.map(({ value, labelKey }) => ({ value, label: t(labelKey) }));
    const selectPlaceholder = t("mcpServers.toolArgumentsForm.selectPlaceholder", { field: field.key });
    return (
      <Select
        items={field.required ? booleanItems : [{ value: null, label: selectPlaceholder }, ...booleanItems]}
        value={control.value ?? null}
        onValueChange={control.onChange}
      >
        <SelectTrigger
          id={control.id}
          aria-invalid={control["aria-invalid"]}
          title={booleanTitle(control.value, t)}
          className="w-full"
        >
          <SelectValue placeholder={selectPlaceholder} />
        </SelectTrigger>
        <SelectContent>
          {!field.required && <SelectItem value={null}>{selectPlaceholder}</SelectItem>}
          <SelectItem value={true}>{t("mcpServers.toolArgumentsForm.boolean.true")}</SelectItem>
          <SelectItem value={false}>{t("mcpServers.toolArgumentsForm.boolean.false")}</SelectItem>
        </SelectContent>
      </Select>
    );
  }

  if (prop.type === "object" || prop.type === "array") {
    return <JsonArgumentControl field={field} prop={prop} control={control} />;
  }

  return (
    <Input
      {...control}
      value={(control.value as string) ?? ""}
      placeholder={prop.description || t("mcpServers.toolArgumentsForm.enterPlaceholder", { field: field.key })}
      className="rounded-lg"
    />
  );
};

const ToolArgumentFields: React.FC<{
  fields: readonly ToolArgumentField[];
  control: Control<ToolArgumentsFormValues>;
  singleInputFallback: boolean;
}> = ({ fields, control, singleInputFallback }) => {
  const { t } = useTranslation();
  if (singleInputFallback) {
    return (
      <FieldGroup>
        <FormField
          control={control}
          name="args.0"
          label={
            <span>
              {t("mcpServers.toolArgumentsForm.singleInput.label")} <span className="text-destructive">*</span>
            </span>
          }
        >
          {(field) => (
            <Input
              {...field}
              value={(field.value as string) ?? ""}
              placeholder={t("mcpServers.toolArgumentsForm.singleInput.placeholder")}
              className="rounded-lg"
            />
          )}
        </FormField>
      </FieldGroup>
    );
  }

  if (fields.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-muted py-6 text-center">
        <div className="mx-auto max-w-sm">
          <h4 className="mb-1 text-sm font-medium text-foreground">
            {t("mcpServers.toolArgumentsForm.noParameters.title")}
          </h4>
          <p className="text-xs text-muted-foreground">{t("mcpServers.toolArgumentsForm.noParameters.description")}</p>
        </div>
      </div>
    );
  }

  return (
    <FieldGroup>
      {fields.map((field, index) => (
        <FormField
          key={`${field.key}-${index}`}
          control={control}
          name={`args.${index}` as const}
          label={argumentLabel(field)}
        >
          {(itemControl) => <ToolArgumentControl field={field} control={itemControl} />}
        </FormField>
      ))}
    </FieldGroup>
  );
};

const callButtonLabel = (isLoading: boolean, hasRun: boolean, t: TFunction): string => {
  if (isLoading) return t("mcpServers.toolArgumentsForm.callButton.calling");
  return hasRun
    ? t("mcpServers.toolArgumentsForm.callButton.callAgain")
    : t("mcpServers.toolArgumentsForm.callButton.call");
};

export const ToolArgumentsForm: React.FC<{
  fields: readonly ToolArgumentField[];
  singleInputFallback: boolean;
  isLoading: boolean;
  hasRun: boolean;
  onRun: (args: Record<string, unknown>) => void;
}> = ({ fields, singleInputFallback, isLoading, hasRun, onRun }) => {
  const { t } = useTranslation();
  const form = useForm<ToolArgumentsFormValues>({
    defaultValues: { args: initialArgumentValues(fields) },
    resolver: toolArgumentsResolver(fields, t),
  });

  const submit = form.handleSubmit((values) => onRun(buildToolCallArguments(fields, values.args)));

  return (
    <TooltipProvider>
      <form onSubmit={submit} className="space-y-3">
        <ToolArgumentFields fields={fields} control={form.control} singleInputFallback={singleInputFallback} />

        <div className="border-t border-border pt-3">
          <Button
            type="button"
            onClick={() => void submit()}
            disabled={isLoading}
            aria-busy={isLoading}
            className="w-full"
          >
            {isLoading && <UiLoadingSpinner className="size-4" />}
            {callButtonLabel(isLoading, hasRun, t)}
          </Button>
        </div>
      </form>
    </TooltipProvider>
  );
};
