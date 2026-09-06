import React from "react";
import { useForm, type Control } from "react-hook-form";
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

const BOOLEAN_ITEMS = [
  { value: true, label: "True" },
  { value: false, label: "False" },
];

const booleanTitle = (value: unknown): string | undefined => {
  if (value === true) return "True";
  if (value === false) return "False";
  return undefined;
};

const JsonArgumentControl: React.FC<{
  field: ToolArgumentField;
  prop: InputSchemaProperty;
  control: FormFieldControlProps<ToolArgumentsFormValues, `args.${number}`>;
}> = ({ field, prop, control }) => {
  const isObject = prop.type === "object";
  const fallbackPlaceholder = isObject ? `Enter JSON object for ${field.key}` : `Enter JSON array for ${field.key}`;
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
        {isObject ? "Provide a valid JSON object." : "Provide a valid JSON array."}
      </p>
    </div>
  );
};

const ToolArgumentControl: React.FC<{
  field: ToolArgumentField;
  control: FormFieldControlProps<ToolArgumentsFormValues, `args.${number}`>;
}> = ({ field, control }) => {
  const prop = resolveSchemaProperty(field.prop);

  if (prop.type === "string" && prop.enum) {
    return (
      <select
        {...control}
        value={(control.value as string) ?? ""}
        className="w-full rounded-lg border border-input bg-transparent px-3 py-2 text-sm shadow-xs transition-colors focus:border-ring focus:ring-3 focus:ring-ring/50 focus:outline-hidden"
      >
        {!field.required && <option value="">Select {field.key}</option>}
        {prop.enum.map((option) => (
          <option key={option} value={option}>
            {option}
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
        placeholder={prop.description || `Enter ${field.key}`}
        className="rounded-lg"
      />
    );
  }

  if (prop.type === "boolean") {
    return (
      <Select
        items={field.required ? BOOLEAN_ITEMS : [{ value: "", label: `Select ${field.key}` }, ...BOOLEAN_ITEMS]}
        value={control.value ?? ""}
        onValueChange={control.onChange}
      >
        <SelectTrigger
          id={control.id}
          aria-invalid={control["aria-invalid"]}
          title={booleanTitle(control.value)}
          className="w-full"
        >
          <SelectValue placeholder={`Select ${field.key}`} />
        </SelectTrigger>
        <SelectContent>
          {!field.required && <SelectItem value="">Select {field.key}</SelectItem>}
          <SelectItem value={true}>True</SelectItem>
          <SelectItem value={false}>False</SelectItem>
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
      placeholder={prop.description || `Enter ${field.key}`}
      className="rounded-lg"
    />
  );
};

const ToolArgumentFields: React.FC<{
  fields: readonly ToolArgumentField[];
  control: Control<ToolArgumentsFormValues>;
  singleInputFallback: boolean;
}> = ({ fields, control, singleInputFallback }) => {
  if (singleInputFallback) {
    return (
      <FieldGroup>
        <FormField
          control={control}
          name="args.0"
          label={
            <span>
              Input <span className="text-destructive">*</span>
            </span>
          }
        >
          {(field) => (
            <Input
              {...field}
              value={(field.value as string) ?? ""}
              placeholder="Enter input for this tool"
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
          <h4 className="mb-1 text-sm font-medium text-foreground">No Parameters Required</h4>
          <p className="text-xs text-muted-foreground">This tool can be called without any input parameters.</p>
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

const callButtonLabel = (isLoading: boolean, hasRun: boolean): string => {
  if (isLoading) return "Calling Tool...";
  return hasRun ? "Call Again" : "Call Tool";
};

export const ToolArgumentsForm: React.FC<{
  fields: readonly ToolArgumentField[];
  singleInputFallback: boolean;
  isLoading: boolean;
  hasRun: boolean;
  onRun: (args: Record<string, unknown>) => void;
}> = ({ fields, singleInputFallback, isLoading, hasRun, onRun }) => {
  const form = useForm<ToolArgumentsFormValues>({
    defaultValues: { args: initialArgumentValues(fields) },
    resolver: toolArgumentsResolver(fields),
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
            {callButtonLabel(isLoading, hasRun)}
          </Button>
        </div>
      </form>
    </TooltipProvider>
  );
};
