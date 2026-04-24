import React, { useState } from "react";
import { Controller, useFieldArray, useFormContext } from "react-hook-form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info, MinusCircle, Plus, X } from "lucide-react";
import { AGENT_FORM_CONFIG, SKILL_FIELD_CONFIG } from "./agent_config";
import CostConfigFields from "./cost_config_fields";

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 inline h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

interface AgentFormFieldsProps {
  showAgentName?: boolean;
  visiblePanels?: string[];
}

interface FieldProps {
  name: string;
  label: string;
  required?: boolean;
  tooltip?: string;
  placeholder?: string;
  type?: string;
  rows?: number;
}

function TextFieldItem({
  name,
  label,
  required,
  tooltip,
  placeholder,
  type,
  rows,
}: FieldProps) {
  const { register, formState } = useFormContext();
  const error = (formState.errors as any)?.[name];
  const inputId = `agent-field-${name}`;
  return (
    <div className="space-y-2">
      <Label htmlFor={inputId}>
        {label}
        {required && <span className="text-destructive"> *</span>}
        {tooltip ? <InfoTip>{tooltip}</InfoTip> : null}
      </Label>
      {type === "textarea" ? (
        <Textarea
          id={inputId}
          rows={rows}
          placeholder={placeholder}
          aria-invalid={!!error}
          {...register(
            name,
            required
              ? { required: `Please enter ${label.toLowerCase()}` }
              : undefined,
          )}
        />
      ) : (
        <Input
          id={inputId}
          placeholder={placeholder}
          aria-invalid={!!error}
          {...register(
            name,
            required
              ? { required: `Please enter ${label.toLowerCase()}` }
              : undefined,
          )}
        />
      )}
      {error && (
        <p className="text-sm text-destructive">{error.message as string}</p>
      )}
    </div>
  );
}

function SwitchFieldItem({ name, label }: { name: string; label: string }) {
  const { control } = useFormContext();
  return (
    <div className="flex items-center justify-between">
      <Label htmlFor={`agent-switch-${name}`}>{label}</Label>
      <Controller
        control={control}
        name={name}
        render={({ field }) => (
          <Switch
            id={`agent-switch-${name}`}
            checked={!!field.value}
            onCheckedChange={field.onChange}
          />
        )}
      />
    </div>
  );
}

/**
 * Comma-separated tag field. Stored as string[] in the form, rendered as a
 * single text input that splits on commas, consistent with the prior antd
 * `getValueFromEvent` behavior.
 */
function CommaListField({
  name,
  label,
  required,
  placeholder,
}: {
  name: string;
  label: string;
  required?: boolean;
  placeholder?: string;
}) {
  const { control, formState } = useFormContext();
  const error = (formState.errors as any)?.[name.split(".")[0]];
  return (
    <div className="space-y-2">
      <Label htmlFor={`agent-list-${name}`}>
        {label}
        {required && <span className="text-destructive"> *</span>}
      </Label>
      <Controller
        control={control}
        name={name}
        rules={
          required
            ? {
                validate: (v) =>
                  Array.isArray(v) && v.length > 0 && v.some((s) => s?.trim())
                    ? true
                    : "Required",
              }
            : undefined
        }
        render={({ field }) => (
          <Input
            id={`agent-list-${name}`}
            placeholder={placeholder}
            value={Array.isArray(field.value) ? field.value.join(", ") : ""}
            onChange={(e) => {
              const raw = e.target.value;
              field.onChange(
                raw
                  .split(",")
                  .map((s) => s.trim())
                  .filter((s) => s !== "" || required !== true),
              );
            }}
          />
        )}
      />
      {error && (
        <p className="text-sm text-destructive">
          {(error as any).message as string}
        </p>
      )}
    </div>
  );
}

function SkillsList() {
  const { control } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: "skills",
  });
  return (
    <div className="space-y-4">
      {fields.map((f, index) => (
        <div
          key={f.id}
          className="space-y-4 p-4 border border-border rounded-md"
        >
          <TextFieldItem
            name={`skills.${index}.id`}
            label={SKILL_FIELD_CONFIG.id.label}
            required={SKILL_FIELD_CONFIG.id.required}
            placeholder={SKILL_FIELD_CONFIG.id.placeholder}
          />
          <TextFieldItem
            name={`skills.${index}.name`}
            label={SKILL_FIELD_CONFIG.name.label}
            required={SKILL_FIELD_CONFIG.name.required}
            placeholder={SKILL_FIELD_CONFIG.name.placeholder}
          />
          <TextFieldItem
            name={`skills.${index}.description`}
            label={SKILL_FIELD_CONFIG.description.label}
            required={SKILL_FIELD_CONFIG.description.required}
            placeholder={SKILL_FIELD_CONFIG.description.placeholder}
            type="textarea"
            rows={SKILL_FIELD_CONFIG.description.rows}
          />
          <CommaListField
            name={`skills.${index}.tags`}
            label={SKILL_FIELD_CONFIG.tags.label}
            required={SKILL_FIELD_CONFIG.tags.required}
            placeholder={SKILL_FIELD_CONFIG.tags.placeholder}
          />
          <CommaListField
            name={`skills.${index}.examples`}
            label={SKILL_FIELD_CONFIG.examples.label}
            placeholder={SKILL_FIELD_CONFIG.examples.placeholder}
          />
          <Button
            type="button"
            variant="ghost"
            className="text-destructive hover:text-destructive"
            onClick={() => remove(index)}
          >
            <MinusCircle className="h-4 w-4" />
            Remove Skill
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        onClick={() =>
          append({ id: "", name: "", description: "", tags: [], examples: [] })
        }
        className="w-full border-dashed"
      >
        <Plus className="h-4 w-4" />
        Add Skill
      </Button>
    </div>
  );
}

function StaticHeadersList() {
  const { control, register } = useFormContext();
  const { fields, append, remove } = useFieldArray({
    control,
    name: "static_headers",
  });
  return (
    <div className="space-y-2">
      {fields.map((f, index) => (
        <div key={f.id} className="flex items-baseline gap-2 mb-2">
          <Input
            placeholder="Header name (e.g. Authorization)"
            className="w-[220px]"
            {...register(`static_headers.${index}.header`, {
              required: "Header name required",
            })}
          />
          <Input
            placeholder="Value (e.g. Bearer token123)"
            className="w-[260px]"
            {...register(`static_headers.${index}.value`, {
              required: "Value required",
            })}
          />
          <button
            type="button"
            onClick={() => remove(index)}
            className="text-destructive"
            aria-label="Remove header"
          >
            <MinusCircle className="h-4 w-4" />
          </button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        onClick={() => append({ header: "", value: "" })}
        className="w-full border-dashed"
      >
        <Plus className="h-4 w-4" />
        Add Static Header
      </Button>
    </div>
  );
}

/**
 * Free-form tag input (replacement for `antd.Select mode="tags"`). Stores a
 * string[] in form state; users type a value and press Enter/comma to add.
 */
function FreeformTagsField({
  name,
  placeholder,
}: {
  name: string;
  placeholder?: string;
}) {
  const { control } = useFormContext();
  const [draft, setDraft] = useState("");
  return (
    <Controller
      control={control}
      name={name}
      render={({ field }) => {
        const values: string[] = Array.isArray(field.value) ? field.value : [];
        const commitDraft = () => {
          const parts = draft
            .split(",")
            .map((s) => s.trim())
            .filter((s) => s && !values.includes(s));
          if (parts.length > 0) {
            field.onChange([...values, ...parts]);
          }
          setDraft("");
        };
        return (
          <div className="space-y-2">
            <Input
              placeholder={placeholder}
              value={draft}
              onChange={(e) => {
                const v = e.target.value;
                if (v.includes(",")) {
                  const parts = v
                    .split(",")
                    .map((s) => s.trim())
                    .filter((s) => s && !values.includes(s));
                  if (parts.length > 0) {
                    field.onChange([...values, ...parts]);
                  }
                  setDraft("");
                } else {
                  setDraft(v);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitDraft();
                }
              }}
              onBlur={commitDraft}
            />
            {values.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {values.map((v) => (
                  <Badge
                    key={v}
                    variant="secondary"
                    className="flex items-center gap-1"
                  >
                    {v}
                    <button
                      type="button"
                      onClick={() =>
                        field.onChange(values.filter((x) => x !== v))
                      }
                      className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                      aria-label={`Remove ${v}`}
                    >
                      <X size={12} />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>
        );
      }}
    />
  );
}

/**
 * Reusable form fields component for agent forms
 * Uses shared configuration from agent_config.ts
 */
const AgentFormFields: React.FC<AgentFormFieldsProps> = ({
  showAgentName = true,
  visiblePanels,
}) => {
  const shouldShow = (key: string) =>
    !visiblePanels || visiblePanels.includes(key);

  const defaultOpen: string[] = ["basic"];

  return (
    <div className="space-y-4">
      {showAgentName && (
        <TextFieldItem
          name="agent_name"
          label="Agent Name"
          required
          tooltip="Unique identifier for the agent"
          placeholder="e.g., customer-support-agent"
        />
      )}

      <Accordion
        type="multiple"
        defaultValue={defaultOpen}
        className="mb-4 space-y-2"
      >
        {/* Basic Information */}
        {shouldShow(AGENT_FORM_CONFIG.basic.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.basic.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.basic.title} (Required)
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                {AGENT_FORM_CONFIG.basic.fields.map((field) => (
                  <TextFieldItem
                    key={field.name}
                    name={field.name}
                    label={field.label}
                    required={field.required}
                    tooltip={field.tooltip}
                    placeholder={field.placeholder}
                    type={field.type === "textarea" ? "textarea" : undefined}
                    rows={field.rows}
                  />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}

        {/* Skills */}
        {shouldShow(AGENT_FORM_CONFIG.skills.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.skills.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.skills.title} (Required)
            </AccordionTrigger>
            <AccordionContent>
              <SkillsList />
            </AccordionContent>
          </AccordionItem>
        )}

        {/* Capabilities */}
        {shouldShow(AGENT_FORM_CONFIG.capabilities.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.capabilities.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.capabilities.title}
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                {AGENT_FORM_CONFIG.capabilities.fields.map((field) => (
                  <SwitchFieldItem
                    key={field.name}
                    name={field.name}
                    label={field.label}
                  />
                ))}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}

        {/* Optional Settings */}
        {shouldShow(AGENT_FORM_CONFIG.optional.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.optional.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.optional.title}
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                {AGENT_FORM_CONFIG.optional.fields.map((field) =>
                  field.type === "switch" ? (
                    <SwitchFieldItem
                      key={field.name}
                      name={field.name}
                      label={field.label}
                    />
                  ) : (
                    <TextFieldItem
                      key={field.name}
                      name={field.name}
                      label={field.label}
                      placeholder={field.placeholder}
                    />
                  ),
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}

        {/* Cost Configuration */}
        {shouldShow(AGENT_FORM_CONFIG.cost.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.cost.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.cost.title}
            </AccordionTrigger>
            <AccordionContent>
              <CostConfigFields />
            </AccordionContent>
          </AccordionItem>
        )}

        {/* LiteLLM Parameters */}
        {shouldShow(AGENT_FORM_CONFIG.litellm.key) && (
          <AccordionItem
            value={AGENT_FORM_CONFIG.litellm.key}
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              {AGENT_FORM_CONFIG.litellm.title}
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                {AGENT_FORM_CONFIG.litellm.fields.map((field) =>
                  field.type === "switch" ? (
                    <SwitchFieldItem
                      key={field.name}
                      name={field.name}
                      label={field.label}
                    />
                  ) : (
                    <TextFieldItem
                      key={field.name}
                      name={field.name}
                      label={field.label}
                      placeholder={field.placeholder}
                    />
                  ),
                )}
              </div>
            </AccordionContent>
          </AccordionItem>
        )}

        {/* Authentication Headers */}
        {shouldShow("auth_headers") && (
          <AccordionItem
            value="auth_headers"
            className="border border-border rounded-md px-3"
          >
            <AccordionTrigger className="py-2 hover:no-underline">
              Authentication Headers
            </AccordionTrigger>
            <AccordionContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>
                    Static Headers
                    <InfoTip>
                      Headers always sent to the backend agent, regardless of
                      the client request. Admin-configured, static wins on
                      conflict.
                    </InfoTip>
                  </Label>
                  <StaticHeadersList />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="agent-extra-headers">
                    Forward Client Headers
                    <InfoTip>
                      Header names to extract from the client&apos;s request
                      and forward to the agent. Type a name and press Enter.
                    </InfoTip>
                  </Label>
                  <FreeformTagsField
                    name="extra_headers"
                    placeholder="e.g. x-api-key, Authorization"
                  />
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        )}
      </Accordion>
    </div>
  );
};

export default AgentFormFields;
