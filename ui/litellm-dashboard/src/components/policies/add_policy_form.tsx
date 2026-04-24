import React, { useState, useEffect, useMemo } from "react";
import { Controller, FormProvider, useForm, useFormContext } from "react-hook-form";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Info, X } from "lucide-react";
import { Policy, PolicyCreateRequest, PolicyUpdateRequest } from "./types";
import { Guardrail } from "../guardrails/types";
import { getResolvedGuardrails, modelAvailableCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface AddPolicyFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onOpenFlowBuilder: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  existingPolicies: Policy[];
  availableGuardrails: Guardrail[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  createPolicy: (accessToken: string, policyData: any) => Promise<any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updatePolicy: (accessToken: string, policyId: string, policyData: any) => Promise<any>;
}

interface PolicyFormValues {
  policy_name: string;
  description: string;
  inherit: string | null;
  guardrails_add: string[];
  guardrails_remove: string[];
  model_condition: string | undefined;
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiSelect (shadcn Select + chip list) — same pattern as AccessGroupBaseForm
// ─────────────────────────────────────────────────────────────────────────────

function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode Picker (Step 1) - shown first when creating a new policy
// The categorical palette (indigo accent) is intentionally kept raw to preserve
// the existing branded look. See docs/DEVIATIONS.md.
// ─────────────────────────────────────────────────────────────────────────────

interface ModePickerProps {
  selected: "simple" | "flow_builder";
  onSelect: (mode: "simple" | "flow_builder") => void;
}

const ModePicker: React.FC<ModePickerProps> = ({ selected, onSelect }) => (
  <div className="flex gap-4 py-2">
    {/* Simple Mode Card */}
    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
    <div
      onClick={() => onSelect("simple")}
      className={`flex-1 rounded-xl cursor-pointer transition-all p-6 border-2 ${
        selected === "simple"
          ? "border-indigo-600 bg-indigo-50"
          : "border-border bg-background"
      }`}
    >
      {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
      <div
        className={`w-10 h-10 rounded-lg flex items-center justify-center mb-4 ${
          selected === "simple" ? "bg-indigo-100" : "bg-muted"
        }`}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke={selected === "simple" ? "#4f46e5" : "#6b7280"}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M8 7h8M8 12h8M8 17h5" />
        </svg>
      </div>
      <div className="text-[15px] font-semibold mb-1">Simple Mode</div>
      <p className="text-sm text-muted-foreground">
        Pick guardrails from a list. All run in parallel.
      </p>
    </div>

    {/* Flow Builder Card */}
    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
    <div
      onClick={() => onSelect("flow_builder")}
      className={`flex-1 rounded-xl cursor-pointer transition-all p-6 border-2 relative ${
        selected === "flow_builder"
          ? "border-indigo-600 bg-indigo-50"
          : "border-border bg-background"
      }`}
    >
      <Badge
        variant="secondary"
        // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
        className="absolute top-3 right-3 bg-purple-100 text-purple-700 text-[10px] font-semibold"
      >
        NEW
      </Badge>
      {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
      <div
        className={`w-10 h-10 rounded-lg flex items-center justify-center mb-4 ${
          selected === "flow_builder" ? "bg-indigo-100" : "bg-muted"
        }`}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke={selected === "flow_builder" ? "#4f46e5" : "#6b7280"}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
      </div>
      <div className="text-[15px] font-semibold mb-1">Flow Builder</div>
      <p className="text-sm text-muted-foreground">
        Define steps, conditions, and error responses.
      </p>
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// Form fields
// ─────────────────────────────────────────────────────────────────────────────

interface PolicyFieldsProps {
  isEditing: boolean;
  modelConditionType: "model" | "regex";
  onModelConditionTypeChange: (t: "model" | "regex") => void;
  policyOptions: { label: string; value: string }[];
  guardrailOptions: { label: string; value: string }[];
  availableModels: string[];
  resolvedGuardrails: string[];
}

function PolicyFields({
  isEditing,
  modelConditionType,
  onModelConditionTypeChange,
  policyOptions,
  guardrailOptions,
  availableModels,
  resolvedGuardrails,
}: PolicyFieldsProps) {
  const { control, register, formState } = useFormContext<PolicyFormValues>();

  return (
    <>
      <div className="space-y-2">
        <Label htmlFor="policy_name">
          Policy Name <span className="text-destructive">*</span>
        </Label>
        <Input
          id="policy_name"
          placeholder="e.g., global-baseline, healthcare-compliance"
          disabled={isEditing}
          aria-invalid={!!formState.errors.policy_name}
          {...register("policy_name", {
            required: "Please enter a policy name",
            pattern: {
              value: /^[a-zA-Z0-9_-]+$/,
              message:
                "Policy name can only contain letters, numbers, hyphens, and underscores",
            },
          })}
        />
        {formState.errors.policy_name && (
          <p className="text-sm text-destructive">
            {formState.errors.policy_name.message as string}
          </p>
        )}
      </div>

      <div className="space-y-2 mt-4">
        <Label htmlFor="policy_description">Description</Label>
        <Textarea
          id="policy_description"
          rows={2}
          placeholder="Describe what this policy does..."
          {...register("description")}
        />
      </div>

      <div className="mt-6 mb-3 flex items-center gap-3">
        <span className="text-sm font-semibold">Inheritance</span>
        <Separator className="flex-1" />
      </div>

      <div className="space-y-2">
        <Label>Inherit From</Label>
        <p className="text-xs text-muted-foreground">
          Inherit guardrails from another policy. The child policy will include
          all guardrails from the parent.
        </p>
        <Controller
          control={control}
          name="inherit"
          render={({ field }) => (
            <div className="flex items-center gap-2">
              <Select
                value={field.value ?? ""}
                onValueChange={(v) => field.onChange(v || null)}
              >
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Select a parent policy (optional)" />
                </SelectTrigger>
                <SelectContent>
                  {policyOptions.length === 0 ? (
                    <div className="py-2 px-3 text-sm text-muted-foreground">
                      No parent policies
                    </div>
                  ) : (
                    policyOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
              {field.value && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => field.onChange(null)}
                >
                  Clear
                </Button>
              )}
            </div>
          )}
        />
      </div>

      <div className="mt-6 mb-3 flex items-center gap-3">
        <span className="text-sm font-semibold">Guardrails</span>
        <Separator className="flex-1" />
      </div>

      <div className="space-y-2">
        <Label>Guardrails to Add</Label>
        <p className="text-xs text-muted-foreground">
          These guardrails will be added to requests matching this policy.
        </p>
        <Controller
          control={control}
          name="guardrails_add"
          render={({ field }) => (
            <MultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={guardrailOptions}
              placeholder="Select guardrails to add"
              emptyText="No guardrails available"
            />
          )}
        />
      </div>

      <div className="space-y-2 mt-4">
        <Label>Guardrails to Remove</Label>
        <p className="text-xs text-muted-foreground">
          These guardrails will be removed from inherited guardrails.
        </p>
        <Controller
          control={control}
          name="guardrails_remove"
          render={({ field }) => (
            <MultiSelect
              value={field.value ?? []}
              onChange={field.onChange}
              options={guardrailOptions}
              placeholder="Select guardrails to remove (from inherited)"
              emptyText="No guardrails available"
            />
          )}
        />
      </div>

      {resolvedGuardrails.length > 0 && (
        <Alert className="mt-4">
          <Info className="h-4 w-4" />
          <AlertTitle>Resolved Guardrails</AlertTitle>
          <AlertDescription>
            <p className="text-muted-foreground mb-2">
              These are the final guardrails that will be applied (including
              inheritance):
            </p>
            <div className="flex flex-wrap gap-1">
              {resolvedGuardrails.map((g) => (
                <Badge
                  key={g}
                  variant="secondary"
                  // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
                  className="bg-blue-100 text-blue-700"
                >
                  {g}
                </Badge>
              ))}
            </div>
          </AlertDescription>
        </Alert>
      )}

      <div className="mt-6 mb-3 flex items-center gap-3">
        <span className="text-sm font-semibold">Conditions (Optional)</span>
        <Separator className="flex-1" />
      </div>

      <Alert className="mb-4">
        <Info className="h-4 w-4" />
        <AlertTitle>Model Scope</AlertTitle>
        <AlertDescription>
          By default, this policy will run on all models. You can optionally
          restrict it to specific models below.
        </AlertDescription>
      </Alert>

      <div className="space-y-2">
        <Label>Model Condition Type</Label>
        <RadioGroup
          value={modelConditionType}
          onValueChange={(v) =>
            onModelConditionTypeChange(v as "model" | "regex")
          }
          className="flex gap-4"
        >
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="model" />
            Select Model
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <RadioGroupItem value="regex" />
            Custom Regex Pattern
          </label>
        </RadioGroup>
      </div>

      <div className="space-y-2 mt-3">
        <Label>
          {modelConditionType === "model"
            ? "Model (Optional)"
            : "Regex Pattern (Optional)"}
        </Label>
        <p className="text-xs text-muted-foreground">
          {modelConditionType === "model"
            ? "Select a specific model to apply this policy to. Leave empty to apply to all models."
            : "Enter a regex pattern to match models (e.g., gpt-4.* or bedrock/.*). Leave empty to apply to all models."}
        </p>
        <Controller
          control={control}
          name="model_condition"
          render={({ field }) =>
            modelConditionType === "model" ? (
              <div className="flex items-center gap-2">
                <Select
                  value={field.value ?? ""}
                  onValueChange={(v) => field.onChange(v || undefined)}
                >
                  <SelectTrigger className="flex-1">
                    <SelectValue placeholder="Leave empty to apply to all models" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.length === 0 ? (
                      <div className="py-2 px-3 text-sm text-muted-foreground">
                        No models available
                      </div>
                    ) : (
                      availableModels.map((m) => (
                        <SelectItem key={m} value={m}>
                          {m}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
                {field.value && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => field.onChange(undefined)}
                  >
                    Clear
                  </Button>
                )}
              </div>
            ) : (
              <Input
                placeholder="Leave empty to apply to all models (e.g., gpt-4.* or bedrock/claude-.*)"
                value={field.value ?? ""}
                onChange={(e) => field.onChange(e.target.value || undefined)}
              />
            )
          }
        />
      </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

const AddPolicyForm: React.FC<AddPolicyFormProps> = ({
  visible,
  onClose,
  onSuccess,
  onOpenFlowBuilder,
  accessToken,
  editingPolicy,
  existingPolicies,
  availableGuardrails,
  createPolicy,
  updatePolicy,
}) => {
  const form = useForm<PolicyFormValues>({
    defaultValues: {
      policy_name: "",
      description: "",
      inherit: null,
      guardrails_add: [],
      guardrails_remove: [],
      model_condition: undefined,
    },
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [resolvedGuardrails, setResolvedGuardrails] = useState<string[]>([]);
  const [modelConditionType, setModelConditionType] = useState<
    "model" | "regex"
  >("model");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [step, setStep] = useState<"pick_mode" | "simple_form">("pick_mode");
  const [selectedMode, setSelectedMode] = useState<"simple" | "flow_builder">(
    "simple",
  );
  const { userId, userRole } = useAuthorized();

  const isEditing = !!editingPolicy?.policy_id;

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (visible && editingPolicy) {
      const modelCondition = editingPolicy.condition?.model;
      const isRegex =
        modelCondition && /[.*+?^${}()|[\]\\]/.test(modelCondition);
      setModelConditionType(isRegex ? "regex" : "model");

      form.reset({
        policy_name: editingPolicy.policy_name,
        description: editingPolicy.description ?? "",
        inherit: editingPolicy.inherit ?? null,
        guardrails_add: editingPolicy.guardrails_add || [],
        guardrails_remove: editingPolicy.guardrails_remove || [],
        model_condition: modelCondition,
      });

      if (editingPolicy.policy_id && accessToken) {
        loadResolvedGuardrails(editingPolicy.policy_id);
      }

      if (editingPolicy.pipeline) {
        onClose();
        onOpenFlowBuilder();
        return;
      }
      setStep("simple_form");
    } else if (visible) {
      form.reset({
        policy_name: "",
        description: "",
        inherit: null,
        guardrails_add: [],
        guardrails_remove: [],
        model_condition: undefined,
      });
      setResolvedGuardrails([]);
      setModelConditionType("model");
      setSelectedMode("simple");
      setStep("pick_mode");
    }
  }, [visible, editingPolicy]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (visible && accessToken) {
      loadAvailableModels();
    }
  }, [visible, accessToken]);

  // Recompute resolved guardrails on form change
  useEffect(() => {
    const subscription = form.watch(() => {
      setResolvedGuardrails(computeResolvedGuardrails());
    });
    return () => subscription.unsubscribe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form.watch, existingPolicies]);

  const loadAvailableModels = async () => {
    if (!accessToken) return;
    try {
      const response = await modelAvailableCall(accessToken, userId, userRole);
      if (response?.data) {
        const models = response.data
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          .map((m: any) => m.id || m.model_name)
          .filter(Boolean);
        setAvailableModels(models);
      }
    } catch (error) {
      console.error("Failed to load available models:", error);
    }
  };

  const loadResolvedGuardrails = async (policyId: string) => {
    if (!accessToken) return;
    try {
      const data = await getResolvedGuardrails(accessToken, policyId);
      setResolvedGuardrails(data.resolved_guardrails || []);
    } catch (error) {
      console.error("Failed to load resolved guardrails:", error);
    }
  };

  const computeResolvedGuardrails = (): string[] => {
    const values = form.getValues();
    const inheritFrom = values.inherit;
    const guardrailsAdd = values.guardrails_add || [];
    const guardrailsRemove = values.guardrails_remove || [];

    const resolved = new Set<string>();

    if (inheritFrom) {
      const parentPolicy = existingPolicies.find(
        (p) => p.policy_name === inheritFrom,
      );
      if (parentPolicy) {
        const parentResolved = resolveParentGuardrails(parentPolicy);
        parentResolved.forEach((g) => resolved.add(g));
      }
    }

    guardrailsAdd.forEach((g: string) => resolved.add(g));
    guardrailsRemove.forEach((g: string) => resolved.delete(g));

    return Array.from(resolved).sort();
  };

  const resolveParentGuardrails = (policy: Policy): string[] => {
    const resolved = new Set<string>();

    if (policy.inherit) {
      const grandparent = existingPolicies.find(
        (p) => p.policy_name === policy.inherit,
      );
      if (grandparent) {
        resolveParentGuardrails(grandparent).forEach((g) => resolved.add(g));
      }
    }
    if (policy.guardrails_add) {
      policy.guardrails_add.forEach((g) => resolved.add(g));
    }
    if (policy.guardrails_remove) {
      policy.guardrails_remove.forEach((g) => resolved.delete(g));
    }
    return Array.from(resolved);
  };

  const handleClose = () => {
    form.reset();
    setStep("pick_mode");
    setSelectedMode("simple");
    onClose();
  };

  const handleModeConfirm = () => {
    if (selectedMode === "flow_builder") {
      onClose();
      onOpenFlowBuilder();
    } else {
      setStep("simple_form");
    }
  };

  const onSubmit = form.handleSubmit(async (values) => {
    try {
      setIsSubmitting(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const data: PolicyCreateRequest | PolicyUpdateRequest = {
        policy_name: values.policy_name,
        description: values.description || undefined,
        inherit: values.inherit || undefined,
        guardrails_add: values.guardrails_add || [],
        guardrails_remove: values.guardrails_remove || [],
        condition: values.model_condition
          ? { model: values.model_condition }
          : undefined,
      };

      if (isEditing && editingPolicy) {
        await updatePolicy(
          accessToken,
          editingPolicy.policy_id,
          data as PolicyUpdateRequest,
        );
        NotificationsManager.success("Policy updated successfully");
      } else {
        await createPolicy(accessToken, data as PolicyCreateRequest);
        NotificationsManager.success("Policy created successfully");
      }

      form.reset();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save policy:", error);
      NotificationsManager.fromBackend(
        "Failed to save policy: " +
          (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setIsSubmitting(false);
    }
  });

  const guardrailOptions = availableGuardrails.map((g) => ({
    label: g.guardrail_name || g.guardrail_id,
    value: g.guardrail_name || g.guardrail_id,
  }));

  const policyOptions = existingPolicies
    .filter((p) => !editingPolicy || p.policy_id !== editingPolicy.policy_id)
    .map((p) => ({
      label: p.policy_name,
      value: p.policy_name,
    }));

  // ── Mode Picker Step ──────────────────────────────────────────────────────
  if (step === "pick_mode") {
    return (
      <Dialog
        open={visible}
        onOpenChange={(o) => (!o ? handleClose() : undefined)}
      >
        <DialogContent className="max-w-[620px]">
          <DialogHeader>
            <DialogTitle>Create New Policy</DialogTitle>
          </DialogHeader>
          <ModePicker selected={selectedMode} onSelect={setSelectedMode} />

          {selectedMode === "flow_builder" && (
            <Alert className="mt-4">
              <Info className="h-4 w-4" />
              <AlertDescription>
                You&apos;ll be redirected to the full-screen Flow Builder to
                design your policy logic visually.
              </AlertDescription>
            </Alert>
          )}

          <div className="flex justify-end gap-2 mt-6">
            <Button variant="secondary" onClick={handleClose}>
              Cancel
            </Button>
            <Button onClick={handleModeConfirm}>
              {selectedMode === "flow_builder"
                ? "Continue to Builder"
                : "Create Policy"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // ── Simple Form Step ──────────────────────────────────────────────────────
  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[700px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {isEditing ? "Edit Policy" : "Create New Policy"}
          </DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <PolicyFields
              isEditing={isEditing}
              modelConditionType={modelConditionType}
              onModelConditionTypeChange={(t) => {
                setModelConditionType(t);
                form.setValue("model_condition", undefined);
              }}
              policyOptions={policyOptions}
              guardrailOptions={guardrailOptions}
              availableModels={availableModels}
              resolvedGuardrails={resolvedGuardrails}
            />

            <div className="flex justify-end space-x-2 mt-4">
              <Button type="button" variant="secondary" onClick={handleClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting
                  ? isEditing
                    ? "Updating..."
                    : "Creating..."
                  : isEditing
                    ? "Update Policy"
                    : "Create Policy"}
              </Button>
            </div>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPolicyForm;
