"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React, { useEffect, useMemo } from "react";
import { useWatch } from "react-hook-form";
import { z } from "zod";
import { englishTranslate } from "@/i18n/language";
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
  groupNameByModel,
  onClose,
  onSubmit,
  saving,
}) => {
  const { t } = useLanguage();
  const modelsAnchor = useComboboxAnchor();
  const selectableStrategies =
    mode === "edit" && initialValue
      ? Array.from(new Set([...availableStrategies, initialValue.routing_strategy]))
      : availableStrategies;
  const strategyItems = selectableStrategies.map((strategy) => ({
    label: strategy === "priority" ? t("Priority") : strategy,
    value: strategy,
  }));

  const reservedNames = useMemo(() => {
    const others = existingGroupNames.filter((n) => n !== initialValue?.group_name);
    return new Set(others.map((n) => n.toLowerCase()));
  }, [existingGroupNames, initialValue]);

  const schema = useMemo(() => {
    // 本地校验保存英文原文，动态详情在字段展示时按当前语言组成。
    const shape = {
      group_name: z
        .string()
        .trim()
        .min(1, "Group name is required")
        .max(
          GROUP_NAME_MAX_LENGTH,
          englishTranslate("Must be {count} characters or fewer", { count: GROUP_NAME_MAX_LENGTH }),
        )
        .refine((value) => !reservedNames.has(value.toLowerCase()), "A group with this name already exists"),
      models: z.array(z.string()).min(1, "Select at least one model"),
      routing_strategy: z.string().min(1, "Strategy is required"),
      routing_strategy_args: z.string(),
      model_priorities: z.array(z.object({ model: z.string(), priority: z.string() })),
    };
    return z.object(shape).superRefine((values, ctx) => {
      if (values.routing_strategy === "priority") return;
      const conflict = modelConflictError(values.models, groupNameByModel);
      if (conflict !== null) {
        ctx.addIssue({ code: "custom", message: conflict, path: ["models"] });
      }
    });
  }, [reservedNames, groupNameByModel]);

  const form = useZodForm(schema, { defaultValues: toRoutingGroupFormValues(initialValue, availableStrategies) });

  useEffect(() => {
    form.reset(toRoutingGroupFormValues(initialValue, availableStrategies));
  }, [open, initialValue, availableStrategies, form]);

  const selectedStrategy = useWatch({ control: form.control, name: "routing_strategy" });
  const selectedModels = useWatch({ control: form.control, name: "models" });

  const handleSubmit = async (values: z.infer<typeof schema>) => {
    const payload = buildRoutingGroupPayload(values);
    if (!payload.ok) {
      // payload.message 仅来自本地纯校验函数，不包含后端错误或用户原文
      form.setError(payload.field, { type: "local-validation", message: payload.message });
      return;
    }
    await onSubmit(payload.group);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>
            {mode === "create" ? t("Create Routing Group") : t("Edit {name}", { name: initialValue?.group_name ?? "" })}
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="group_name"
              renderError={(error) =>
                error.type === "too_big"
                  ? t("Must be {count} characters or fewer", { count: GROUP_NAME_MAX_LENGTH })
                  : error.message && t(error.message)
              }
              label={t("Routing Group Name")}
              description={t(
                "Use this name as the model in API calls. LiteLLM routes the request to one of the group's models.",
              )}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="fast-chat" disabled={mode === "edit"} />}
            </FormField>

            <FormField
              control={form.control}
              name="models"
              renderError={(error) =>
                error.type === "custom"
                  ? modelConflictError(selectedModels, groupNameByModel, t) ?? error.message
                  : error.message && t(error.message)
              }
              label={t("Models")}
              description={
                selectedStrategy === "priority"
                  ? t(
                      "Models from your model list that this group routes between. Models can belong to multiple priority groups.",
                    )
                  : t(
                      "Models from your model list that this group routes between. A model can belong to one non-priority group.",
                    )
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
                            placeholder={t("Select models")}
                          />
                        </>
                      )}
                    </ComboboxValue>
                  </ComboboxChips>
                  <ComboboxContent anchor={modelsAnchor}>
                    <ComboboxEmpty>{t("No models found")}</ComboboxEmpty>
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
              renderError={(error) => error.message && t(error.message)}
              label={t("Routing Strategy")}
              description={
                selectedStrategy === "priority"
                  ? t(
                      "Lower priorities are tried first. Models with the same priority share traffic. Applies only when calling this group.",
                    )
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
                    <SelectValue placeholder={t("Select strategy")} />
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
                renderError={(error) =>
                  error.type === "local-validation" && error.message ? t(error.message) : error.message
                }
                label={t("Model Priorities")}
                description={t(
                  "Use 1 for your first choice, 2 for your next choice, and so on. Unavailable models are skipped.",
                )}
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <div id={id} className="space-y-2" role="group" aria-label={t("Model priorities")}>
                    {value.length === 0 && (
                      <p className="text-sm text-muted-foreground">{t("Select models to set priorities")}</p>
                    )}
                    {value.map((entry, index) => (
                      <div key={entry.model} className="flex items-center justify-between gap-3">
                        <label htmlFor={`${id}-${index}`} className="min-w-0 flex-1 break-words text-sm">
                          {entry.model}
                          {!selectedModels.includes(entry.model) && (
                            <span className="block text-xs text-destructive">{t("Model is not selected")}</span>
                          )}
                        </label>
                        <Input
                          id={`${id}-${index}`}
                          aria-label={t("Priority for {model}", { model: String(entry.model) })}
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
                            aria-label={t("Remove priority for {model}", { model: String(entry.model) })}
                            onClick={() => onChange(value.filter((_, itemIndex) => itemIndex !== index))}
                          >
                            {t("Remove")}
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
                renderError={(error) =>
                  error.type === "local-validation" && error.message ? t(error.message) : error.message
                }
                label={t("Strategy Arguments (JSON)")}
                description={
                  t("Example:") + " " + (ARGS_EXAMPLES[selectedStrategy]?.replace(/^Example: /, "") ?? '{ "ttl": 60 }')
                }
              >
                {({ ref, ...field }) => (
                  <Textarea {...field} ref={ref} rows={4} placeholder='{ "ttl": 3600 }' className="font-mono text-xs" />
                )}
              </FormField>
            )}

            <p className="text-xs text-muted-foreground">
              {selectedStrategy === "priority"
                ? t("Direct requests to a member model keep their existing routing behavior.")
                : t("Models outside non-priority groups use the proxy's top-level routing strategy.")}
            </p>
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            {t("Cancel")}
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={saving} aria-busy={saving}>
            {mode === "create" ? t("Create Routing Group") : t("Save Changes")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default RoutingGroupModal;
