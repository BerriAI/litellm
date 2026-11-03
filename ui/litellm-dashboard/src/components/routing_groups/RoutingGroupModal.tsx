"use client";

import React, { useEffect, useMemo } from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import { useWatch } from "react-hook-form";
import { z } from "zod";
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
  STRATEGIES_WITH_ARGS,
  argsForStrategy,
  buildRoutingGroupPayload,
  prioritiesForModels,
  toRoutingGroupFormValues,
} from "./routingGroupPayload";
import type { RoutingGroup } from "./types";
import { modelConflictError } from "./modelOwnership";
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
  groupNameByModel: Record<string, string>;
  onClose: () => void;
  onSubmit: (group: RoutingGroup) => Promise<void> | void;
  saving?: boolean;
}

const ARGS_EXAMPLES: Record<string, ParseKeys> = {
  "latency-based-routing": "routingGroups.modal.latencyArgsExample",
};

const RoutingGroupModal: React.FC<RoutingGroupModalProps> = ({
  open,
  mode,
  initialValue,
  availableStrategies,
  strategyDescriptions,
  modelOptions,
  existingGroupNames,
  groupNameByModel,
  onClose,
  onSubmit,
  saving,
}) => {
  const { t } = useTranslation();
  const modelsAnchor = useComboboxAnchor();
  const selectableStrategies =
    mode === "edit" && initialValue
      ? Array.from(new Set([...availableStrategies, initialValue.routing_strategy]))
      : availableStrategies;
  const strategyItems = selectableStrategies.map((strategy) => ({
    label: strategy === "priority" ? t("routingGroups.strategies.priority") : strategy,
    value: strategy,
  }));

  const reservedNames = useMemo(() => {
    const others = existingGroupNames.filter((n) => n !== initialValue?.group_name);
    return new Set(others.map((n) => n.toLowerCase()));
  }, [existingGroupNames, initialValue]);

  const schema = useMemo(() => {
    const shape = {
      group_name: z
        .string()
        .trim()
        .min(1, t("routingGroups.modal.validation.nameRequired"))
        .max(GROUP_NAME_MAX_LENGTH, t("routingGroups.modal.validation.nameTooLong", { max: GROUP_NAME_MAX_LENGTH }))
        .refine((value) => !reservedNames.has(value.toLowerCase()), t("routingGroups.modal.validation.nameTaken")),
      models: z.array(z.string()).min(1, t("routingGroups.modal.validation.modelsRequired")),
      routing_strategy: z.string().min(1, t("routingGroups.modal.validation.strategyRequired")),
      routing_strategy_args: z.string(),
      model_priorities: z.array(z.object({ model: z.string(), priority: z.string() })),
    };
    return z.object(shape).superRefine((values, ctx) => {
      if (values.routing_strategy === "priority") return;
      const conflict = modelConflictError(values.models, groupNameByModel, t);
      if (conflict !== null) {
        ctx.addIssue({ code: "custom", message: conflict, path: ["models"] });
      }
    });
  }, [reservedNames, groupNameByModel, t]);

  const form = useZodForm(schema, { defaultValues: toRoutingGroupFormValues(initialValue, availableStrategies) });

  useEffect(() => {
    form.reset(toRoutingGroupFormValues(initialValue, availableStrategies));
  }, [open, initialValue, availableStrategies, form]);

  const selectedStrategy = useWatch({ control: form.control, name: "routing_strategy" });
  const selectedModels = useWatch({ control: form.control, name: "models" });

  const handleSubmit = async (values: z.infer<typeof schema>) => {
    const payload = buildRoutingGroupPayload(values, t);
    if (!payload.ok) {
      form.setError(payload.field, { message: payload.message });
      return;
    }
    await onSubmit(payload.group);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {mode === "create"
              ? t("routingGroups.modal.createTitle")
              : t("routingGroups.modal.editTitle", { name: initialValue?.group_name ?? "" })}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="group_name"
              label={t("routingGroups.modal.groupName")}
              description={t("routingGroups.modal.groupNameDescription")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="fast-chat" disabled={mode === "edit"} />}
            </FormField>

            <FormField
              control={form.control}
              name="models"
              label={t("routingGroups.modal.models")}
              description={
                selectedStrategy === "priority"
                  ? t("routingGroups.modal.modelsPriorityDescription")
                  : t("routingGroups.modal.modelsDescription")
              }
            >
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                <Combobox
                  multiple
                  items={modelOptions}
                  value={value}
                  onValueChange={(models: string[]) => {
                    onChange(models);
                    form.setValue("model_priorities", prioritiesForModels(models, form.getValues("model_priorities")));
                  }}
                >
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
                            placeholder={t("routingGroups.modal.modelsPlaceholder")}
                          />
                        </>
                      )}
                    </ComboboxValue>
                  </ComboboxChips>
                  <ComboboxContent anchor={modelsAnchor}>
                    <ComboboxEmpty>{t("routingGroups.modal.noModelsFound")}</ComboboxEmpty>
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
              label={t("routingGroups.modal.routingStrategy")}
              description={
                selectedStrategy === "priority"
                  ? t("routingGroups.modal.priorityStrategyDescription")
                  : strategyDescriptions[selectedStrategy]
              }
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
                    <SelectValue placeholder={t("routingGroups.modal.strategyPlaceholder")} />
                  </SelectTrigger>
                  <SelectContent>
                    {strategyItems.map((strategy) => (
                      <SelectItem key={strategy.value} value={strategy.value}>
                        {strategy.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>

            {selectedStrategy === "priority" && (
              <FormField
                control={form.control}
                name="model_priorities"
                label={t("routingGroups.modal.modelPriorities")}
                description={t("routingGroups.modal.modelPrioritiesDescription")}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <div
                    id={id}
                    className="space-y-2"
                    role="group"
                    aria-label={t("routingGroups.modal.modelPrioritiesAriaLabel")}
                  >
                    {value.length === 0 && (
                      <p className="text-sm text-muted-foreground">
                        {t("routingGroups.modal.selectModelsForPriorities")}
                      </p>
                    )}
                    {value.map((entry, index) => (
                      <div key={entry.model} className="flex items-center justify-between gap-3">
                        <label htmlFor={`${id}-${index}`} className="min-w-0 flex-1 break-words text-sm">
                          {entry.model}
                          {!selectedModels.includes(entry.model) && (
                            <span className="block text-xs text-destructive">
                              {t("routingGroups.modal.modelNotSelected")}
                            </span>
                          )}
                        </label>
                        <Input
                          id={`${id}-${index}`}
                          aria-label={t("routingGroups.modal.priorityFor", { model: entry.model })}
                          aria-invalid={ariaInvalid}
                          aria-describedby={ariaDescribedBy}
                          inputMode="numeric"
                          className="w-24"
                          value={entry.priority}
                          onChange={(event) =>
                            onChange(
                              value.map((item, itemIndex) =>
                                itemIndex === index ? { ...item, priority: event.target.value } : item,
                              ),
                            )
                          }
                        />
                        {!selectedModels.includes(entry.model) && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            aria-label={t("routingGroups.modal.removePriorityFor", { model: entry.model })}
                            onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
                          >
                            {t("routingGroups.modal.remove")}
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </FormField>
            )}

            {STRATEGIES_WITH_ARGS.has(selectedStrategy) && (
              <FormField
                control={form.control}
                name="routing_strategy_args"
                label={t("routingGroups.modal.strategyArgs")}
                description={t(ARGS_EXAMPLES[selectedStrategy] ?? "routingGroups.modal.defaultArgsExample")}
              >
                {({ ref, ...field }) => (
                  <Textarea {...field} ref={ref} rows={4} placeholder='{ "ttl": 3600 }' className="font-mono text-xs" />
                )}
              </FormField>
            )}

            <p className="text-xs text-muted-foreground">
              {selectedStrategy === "priority"
                ? t("routingGroups.modal.priorityFootnote")
                : t("routingGroups.modal.strategyFootnote")}
            </p>
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("routingGroups.modal.cancel")}
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={saving} aria-busy={saving}>
            {mode === "create" ? t("routingGroups.modal.createGroup") : t("routingGroups.modal.saveChanges")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default RoutingGroupModal;
