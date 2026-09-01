"use client";

import React, { useEffect, useMemo } from "react";
import { useWatch } from "react-hook-form";
import { z } from "zod/v4";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
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
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { useZodForm } from "@/lib/forms/useZodForm";
import {
  GROUP_NAME_MAX_LENGTH,
  GROUP_NAME_PATTERN,
  STRATEGIES_WITH_ARGS,
  argsForStrategy,
  buildRoutingGroupPayload,
  toRoutingGroupFormValues,
} from "./routingGroupPayload";
import type { RoutingGroup } from "./types";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface RoutingGroupModalProps {
  open: boolean;
  mode: "create" | "edit";
  initialValue: RoutingGroup | null;
  availableStrategies: string[];
  strategyDescriptions: Record<string, string>;
  modelOptions: string[];
  existingGroupNames: string[];
  onClose: () => void;
  onSubmit: (group: RoutingGroup) => Promise<void> | void;
  saving?: boolean;
}

const ARGS_EXAMPLES: Record<string, string> = {
  "latency-based-routing": 'Example: { "ttl": 3600, "lowest_latency_buffer": 0 }',
};

const RoutingGroupModal: React.FC<RoutingGroupModalProps> = ({
  open,
  mode,
  initialValue,
  availableStrategies,
  strategyDescriptions,
  modelOptions,
  existingGroupNames,
  onClose,
  onSubmit,
  saving,
}) => {
  const modelsAnchor = useComboboxAnchor();
  const strategyItems = availableStrategies.map((strategy) => ({ label: strategy, value: strategy }));

  const reservedNames = useMemo(() => {
    const others = existingGroupNames.filter((n) => n !== initialValue?.group_name);
    return new Set(others.map((n) => n.toLowerCase()));
  }, [existingGroupNames, initialValue]);

  const schema = useMemo(() => {
    const shape = {
      group_name: z
        .string()
        .min(1, "Group name is required")
        .max(GROUP_NAME_MAX_LENGTH, `Must be ${GROUP_NAME_MAX_LENGTH} characters or fewer`)
        .regex(GROUP_NAME_PATTERN, "Only letters, numbers, dot, underscore, and dash are allowed")
        .refine((value) => !reservedNames.has(value.trim().toLowerCase()), "A group with this name already exists"),
      models: z.array(z.string()).min(1, "Select at least one model"),
      routing_strategy: z.string().min(1, "Strategy is required"),
      routing_strategy_args: z.string(),
    };
    return z.object(shape);
  }, [reservedNames]);

  const form = useZodForm(schema, { defaultValues: toRoutingGroupFormValues(initialValue, availableStrategies) });

  useEffect(() => {
    form.reset(toRoutingGroupFormValues(initialValue, availableStrategies));
  }, [open, initialValue, availableStrategies, form]);

  const selectedStrategy = useWatch({ control: form.control, name: "routing_strategy" });

  const handleSubmit = async (values: z.infer<typeof schema>) => {
    const payload = buildRoutingGroupPayload(values);
    if (!payload.ok) {
      form.setError("routing_strategy_args", { message: payload.argsError });
      return;
    }
    await onSubmit(payload.group);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {mode === "create" ? "Create Routing Group" : `Edit ${initialValue?.group_name ?? ""}`}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="group_name"
              label="Group Name"
              description="Use this name as the model in API calls — LiteLLM routes the request to one of the group's models."
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="fast-chat" disabled={mode === "edit"} />}
            </FormField>

            <FormField
              control={form.control}
              name="models"
              label="Models"
              description="Models from your model list that this group routes between."
            >
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Combobox multiple items={modelOptions} value={value} onValueChange={onChange}>
                  <ComboboxChips render={<div ref={modelsAnchor} />}>
                    <ComboboxValue>
                      {(selected: string[]) => (
                        <>
                          {selected.map((model) => (
                            <ComboboxChip key={model} aria-label={model}>
                              {model}
                            </ComboboxChip>
                          ))}
                          <ComboboxChipsInput
                            id={id}
                            aria-invalid={ariaInvalid}
                            aria-describedby={ariaDescribedBy}
                            placeholder="Select models"
                          />
                        </>
                      )}
                    </ComboboxValue>
                  </ComboboxChips>
                  <ComboboxContent anchor={modelsAnchor}>
                    <ComboboxEmpty>No models found</ComboboxEmpty>
                    <ComboboxList>
                      {(model: string) => (
                        <ComboboxItem key={model} value={model}>
                          {model}
                        </ComboboxItem>
                      )}
                    </ComboboxList>
                  </ComboboxContent>
                </Combobox>
              )}
            </FormField>

            <FormField
              control={form.control}
              name="routing_strategy"
              label="Routing Strategy"
              description={strategyDescriptions[selectedStrategy]}
            >
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Select
                  items={strategyItems}
                  value={value}
                  onValueChange={(next: string | null) => {
                    onChange(next ?? "");
                    form.setValue(
                      "routing_strategy_args",
                      argsForStrategy(next ?? "", form.getValues("routing_strategy_args")),
                    );
                  }}
                >
                  <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
                    <SelectValue placeholder="Select strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableStrategies.map((strategy) => (
                      <SelectItem key={strategy} value={strategy}>
                        {strategy}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            {STRATEGIES_WITH_ARGS.has(selectedStrategy) && (
              <FormField
                control={form.control}
                name="routing_strategy_args"
                label="Strategy Arguments (JSON)"
                description={ARGS_EXAMPLES[selectedStrategy] ?? 'Example: { "ttl": 60 }'}
              >
                {({ ref, ...field }) => (
                  <Textarea {...field} ref={ref} rows={4} placeholder='{ "ttl": 3600 }' className="font-mono text-xs" />
                )}
              </FormField>
            )}

            <p className="text-xs text-muted-foreground">
              Models not claimed by an explicit group fall through to the proxy&apos;s top-level routing strategy.
            </p>
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={saving} aria-busy={saving}>
            {mode === "create" ? "Create Group" : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default RoutingGroupModal;
