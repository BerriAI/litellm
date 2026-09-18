"use client";

import * as React from "react";
import { ChevronRight, CircleHelp } from "lucide-react";
import {
  Controller,
  useFormContext,
  type FieldPath,
  type RegisterOptions,
  type ControllerProps,
  type ControllerRenderProps,
} from "react-hook-form";

import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";

export interface AgentSkillFormValue {
  id?: string;
  name?: string;
  description?: string;
  tags?: string[];
  examples?: string[];
}

export interface StaticHeaderFormValue {
  header?: string;
  value?: string;
}

export interface McpServerSelection {
  servers?: string[];
  accessGroups?: string[];
  toolsets?: string[];
}

export type AgentFormFieldValue =
  | string
  | number
  | boolean
  | string[]
  | AgentSkillFormValue[]
  | StaticHeaderFormValue[]
  | McpServerSelection
  | Record<string, string[]>
  | null
  | undefined;

export interface AgentFormValues {
  [credentialKey: string]: AgentFormFieldValue;
  agent_name?: string;
  name?: string;
  display_name?: string;
  description?: string;
  url?: string;
  version?: string;
  protocolVersion?: string;
  skills?: AgentSkillFormValue[];
  streaming?: boolean;
  pushNotifications?: boolean;
  stateTransitionHistory?: boolean;
  iconUrl?: string;
  documentationUrl?: string;
  supportsAuthenticatedExtendedCard?: boolean;
  model?: string;
  make_public?: boolean;
  cost_per_query?: string | number;
  input_cost_per_token?: string | number;
  output_cost_per_token?: string | number;
  static_headers?: StaticHeaderFormValue[];
  extra_headers?: string[];
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  session_tpm_limit?: number | null;
  session_rpm_limit?: number | null;
  team_id?: string;
  guardrails?: string[];
  entitlement_models?: string[];
  entitlement_agents?: string[];
  allowed_mcp_servers_and_groups?: McpServerSelection;
  mcp_tool_permissions?: Record<string, string[]>;
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  enable_tracing?: boolean;
}

export type AgentFieldName = FieldPath<AgentFormValues>;

export const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

export type AgentFieldControlProps = ControllerRenderProps<AgentFormValues, AgentFieldName> & {
  id: string;
  "aria-invalid": true | undefined;
  "aria-describedby": string | undefined;
};

export interface AgentRequestPayload {
  [key: string]: unknown;
  agent_name?: string;
  agent_card_params?: Record<string, unknown>;
  litellm_params?: Record<string, unknown>;
  object_permission?: Record<string, unknown>;
}

interface AgentFormFieldProps {
  name: AgentFieldName;
  label?: React.ReactNode;
  description?: React.ReactNode;
  defaultValue?: AgentFormFieldValue;
  rules?: Omit<RegisterOptions<AgentFormValues, AgentFieldName>, "valueAsNumber" | "valueAsDate" | "setValueAs">;
  className?: string;
  children: (control: AgentFieldControlProps) => React.ReactNode;
}

export const AgentFormField = ({
  name,
  label,
  description,
  defaultValue,
  rules,
  className,
  children,
}: AgentFormFieldProps) => {
  const { control } = useFormContext<AgentFormValues>();
  const reactId = React.useId();
  const controlId = `${reactId}-control`;
  const descriptionId = `${reactId}-description`;
  const errorId = `${reactId}-error`;

  const renderField: ControllerProps<AgentFormValues, AgentFieldName>["render"] = ({ field, fieldState }) => {
    const invalid = fieldState.error !== undefined;
    const describedBy =
      [description !== undefined ? descriptionId : undefined, invalid ? errorId : undefined]
        .filter((id): id is string => id !== undefined)
        .join(" ") || undefined;

    return (
      <Field data-invalid={invalid || undefined} className={className}>
        {label !== undefined && <FieldLabel htmlFor={controlId}>{label}</FieldLabel>}
        {children({
          ...field,
          id: controlId,
          "aria-invalid": invalid || undefined,
          "aria-describedby": describedBy,
        })}
        {description !== undefined && <FieldDescription id={descriptionId}>{description}</FieldDescription>}
        <FieldError id={errorId} errors={[fieldState.error]} />
      </Field>
    );
  };

  return <Controller control={control} name={name} defaultValue={defaultValue} rules={rules} render={renderField} />;
};

export interface CollapsiblePanelsState {
  readonly openPanels: readonly string[];
  readonly mountedPanels: readonly string[];
  readonly toggle: (panelKey: string) => void;
}

export const useCollapsiblePanels = (initiallyOpen: readonly string[]): CollapsiblePanelsState => {
  const [openPanels, setOpenPanels] = React.useState<readonly string[]>(initiallyOpen);
  const [mountedPanels, setMountedPanels] = React.useState<readonly string[]>(initiallyOpen);

  const toggle = React.useCallback((panelKey: string) => {
    setOpenPanels((current) =>
      current.includes(panelKey) ? current.filter((key) => key !== panelKey) : [...current, panelKey],
    );
    setMountedPanels((current) => (current.includes(panelKey) ? current : [...current, panelKey]));
  }, []);

  return { openPanels, mountedPanels, toggle };
};

interface AgentFormPanelProps {
  panelKey: string;
  title: string;
  panels: CollapsiblePanelsState;
  children: React.ReactNode;
}

export const AgentFormPanel = ({ panelKey, title, panels, children }: AgentFormPanelProps) => (
  <Collapsible
    open={panels.openPanels.includes(panelKey)}
    onOpenChange={() => panels.toggle(panelKey)}
    className="border-b border-border last:border-b-0"
  >
    <CollapsibleTrigger className="group flex w-full items-center gap-2 py-3 text-left text-sm font-medium text-foreground">
      <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
      {title}
    </CollapsibleTrigger>
    <CollapsibleContent keepMounted>
      {panels.mountedPanels.includes(panelKey) && <FieldGroup className="pt-1 pb-5">{children}</FieldGroup>}
    </CollapsibleContent>
  </Collapsible>
);

export const omitFieldValues = (values: AgentFormValues, names: readonly string[]): AgentFormValues =>
  Object.fromEntries(Object.entries(values).filter(([key]) => !names.includes(key)));

interface AgentNumberInputProps extends Omit<AgentFieldControlProps, "value" | "onChange" | "ref"> {
  value: AgentFormFieldValue;
  onChange: (value: number | null) => void;
  inputRef: AgentFieldControlProps["ref"];
  min?: number;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export const AgentNumberInput = ({ value, onChange, onBlur, inputRef, min, ...props }: AgentNumberInputProps) => (
  <Input
    {...props}
    ref={inputRef}
    type="number"
    step="any"
    value={typeof value === "number" ? value : ""}
    onWheel={(event) => event.currentTarget.blur()}
    onChange={(event) => {
      const raw = event.target.valueAsNumber;
      onChange(Number.isNaN(raw) ? null : raw);
    }}
    onBlur={() => {
      if (min !== undefined && typeof value === "number" && value < min) onChange(min);
      onBlur();
    }}
  />
);

export interface AgentSelectOption {
  label: string;
  value: string;
}

const matchesQuery = (option: AgentSelectOption, query: string): boolean =>
  option.label.toLowerCase().includes(query.trim().toLowerCase());

const TAG_SEPARATOR = ",";

interface AgentTagsInputProps {
  id: string;
  options?: readonly AgentSelectOption[];
  value: string[];
  onValueChange: (value: string[]) => void;
  placeholder?: string;
  emptyText?: string;
  "aria-invalid"?: true | undefined;
  "aria-describedby"?: string | undefined;
}

export const AgentTagsInput = ({
  id,
  options = [],
  value,
  onValueChange,
  placeholder,
  emptyText = "No matching options",
  ...props
}: AgentTagsInputProps) => {
  const anchor = useComboboxAnchor();
  const [query, setQuery] = React.useState("");
  const pendingRef = React.useRef("");

  const selected = value.map((tag) => options.find((option) => option.value === tag) ?? { label: tag, value: tag });
  const pending = query.trim();
  const items =
    pending.length > 0 && !options.some((option) => option.value === pending)
      ? [{ label: pending, value: pending }, ...options]
      : [...options];

  const setPending = (next: string) => {
    pendingRef.current = next;
    setQuery(next);
  };

  const addTags = (tags: readonly string[]) => {
    const additions = tags
      .map((tag) => tag.trim())
      .filter(Boolean)
      .filter((tag, index, all) => all.indexOf(tag) === index && !value.includes(tag));
    if (additions.length > 0) onValueChange([...value, ...additions]);
  };

  const handleInputValueChange = (next: string, details: { reason: string }) => {
    if (details.reason === "input-clear") {
      const committed = pendingRef.current;
      setPending("");
      addTags([committed]);
      return;
    }
    const parts = next.split(TAG_SEPARATOR);
    setPending(parts[parts.length - 1] ?? "");
    addTags(parts.slice(0, -1));
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== "Enter" || event.currentTarget.getAttribute("aria-activedescendant")) return;
    event.preventDefault();
    const committed = pendingRef.current;
    setPending("");
    addTags([committed]);
  };

  return (
    <Combobox
      multiple
      items={items}
      value={selected}
      onValueChange={(next: AgentSelectOption[]) => {
        setPending("");
        onValueChange(next.map((option) => option.value));
      }}
      inputValue={query}
      onInputValueChange={handleInputValueChange}
      isItemEqualToValue={(option: AgentSelectOption, other: AgentSelectOption) => option.value === other.value}
      itemToStringLabel={(option: AgentSelectOption) => option.label}
      filter={matchesQuery}
      openOnInputClick
    >
      <ComboboxChips render={<div ref={anchor} />} className="min-h-8 py-1 text-sm">
        <ComboboxValue>
          {(chips: AgentSelectOption[]) => (
            <>
              {chips.map((option) => (
                <ComboboxChip key={option.value} aria-label={option.label}>
                  {option.label}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput
                id={id}
                placeholder={placeholder}
                className="min-w-24"
                onKeyDown={handleKeyDown}
                {...props}
              />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: AgentSelectOption) => (
            <ComboboxItem key={option.value} value={option} title={option.label}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};

interface AgentMultiSelectProps {
  id: string;
  options: readonly AgentSelectOption[];
  value: string[];
  onValueChange: (value: string[]) => void;
  placeholder?: string;
  emptyText?: string;
  "aria-invalid"?: true | undefined;
  "aria-describedby"?: string | undefined;
}

export const AgentMultiSelect = ({
  id,
  options,
  value,
  onValueChange,
  placeholder,
  emptyText = "No matching options",
  ...props
}: AgentMultiSelectProps) => {
  const anchor = useComboboxAnchor();
  const items = [...options];
  const selected = value.map((item) => items.find((option) => option.value === item) ?? { label: item, value: item });

  return (
    <Combobox
      multiple
      items={items}
      value={selected}
      onValueChange={(next: AgentSelectOption[]) => onValueChange(next.map((option) => option.value))}
      isItemEqualToValue={(option: AgentSelectOption, other: AgentSelectOption) => option.value === other.value}
      itemToStringLabel={(option: AgentSelectOption) => option.label}
      filter={matchesQuery}
      openOnInputClick
    >
      <ComboboxChips render={<div ref={anchor} />} className="min-h-8 py-1 text-sm">
        <ComboboxValue>
          {(chips: AgentSelectOption[]) => (
            <>
              {chips.map((option) => (
                <ComboboxChip key={option.value} aria-label={option.label}>
                  {option.label}
                </ComboboxChip>
              ))}
              <ComboboxChipsInput id={id} placeholder={placeholder} className="min-w-24" {...props} />
            </>
          )}
        </ComboboxValue>
      </ComboboxChips>
      <ComboboxContent anchor={anchor}>
        <ComboboxEmpty>{emptyText}</ComboboxEmpty>
        <ComboboxList>
          {(option: AgentSelectOption) => (
            <ComboboxItem key={option.value} value={option} title={option.label}>
              {option.label}
            </ComboboxItem>
          )}
        </ComboboxList>
      </ComboboxContent>
    </Combobox>
  );
};
