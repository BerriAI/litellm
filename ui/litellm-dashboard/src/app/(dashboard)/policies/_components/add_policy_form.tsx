import React, { useState, useEffect } from "react";
import { z } from "zod/v4";
import { Policy, PolicyCreateRequest, PolicyUpdateRequest } from "@/components/policies/types";
import { Guardrail } from "@/components/guardrails/types";
import { getResolvedGuardrails, modelAvailableCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { CircleHelp, Info } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";

interface AddPolicyFormProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  onOpenFlowBuilder: () => void;
  accessToken: string | null;
  editingPolicy?: Policy | null;
  existingPolicies: Policy[];
  availableGuardrails: Guardrail[];
  createPolicy: (accessToken: string, policyData: any) => Promise<any>;
  updatePolicy: (accessToken: string, policyId: string, policyData: any) => Promise<any>;
}

type ModelConditionType = "model" | "regex";

const policyShape = {
  policy_name: z
    .string()
    .min(1, "Please enter a policy name")
    .regex(/^[a-zA-Z0-9_-]+$/, "Policy name can only contain letters, numbers, hyphens, and underscores"),
  description: z.string(),
  inherit: z.string(),
  guardrails_add: z.array(z.string()),
  guardrails_remove: z.array(z.string()),
  model_condition: z.string(),
};

const policySchema = z.object(policyShape);

type PolicyFormValues = z.infer<typeof policySchema>;

const EMPTY_VALUES: PolicyFormValues = {
  policy_name: "",
  description: "",
  inherit: "",
  guardrails_add: [],
  guardrails_remove: [],
  model_condition: "",
};

const toFormValues = (policy: Policy): PolicyFormValues => ({
  policy_name: policy.policy_name,
  description: policy.description ?? "",
  inherit: policy.inherit ?? "",
  guardrails_add: policy.guardrails_add || [],
  guardrails_remove: policy.guardrails_remove || [],
  model_condition: policy.condition?.model ?? "",
});

const buildPolicyRequest = (values: PolicyFormValues): PolicyCreateRequest | PolicyUpdateRequest => ({
  policy_name: values.policy_name,
  description: values.description || undefined,
  inherit: values.inherit || undefined,
  guardrails_add: values.guardrails_add,
  guardrails_remove: values.guardrails_remove,
  condition: values.model_condition ? { model: values.model_condition } : undefined,
});

const parentGuardrails = (policy: Policy, existingPolicies: Policy[]): string[] => {
  const inherited = policy.inherit
    ? (() => {
        const grandparent = existingPolicies.find((candidate) => candidate.policy_name === policy.inherit);
        return grandparent ? parentGuardrails(grandparent, existingPolicies) : [];
      })()
    : [];
  const resolved = new Set<string>([...inherited, ...(policy.guardrails_add ?? [])]);
  (policy.guardrails_remove ?? []).forEach((guardrail) => resolved.delete(guardrail));
  return Array.from(resolved);
};

const resolveGuardrails = (values: PolicyFormValues, existingPolicies: Policy[]): string[] => {
  const parentPolicy = values.inherit
    ? existingPolicies.find((policy) => policy.policy_name === values.inherit)
    : undefined;
  const resolved = new Set<string>([
    ...(parentPolicy ? parentGuardrails(parentPolicy, existingPolicies) : []),
    ...values.guardrails_add,
  ]);
  values.guardrails_remove.forEach((guardrail) => resolved.delete(guardrail));
  return Array.from(resolved).sort();
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const SectionHeading: React.FC<{ label: string }> = ({ label }) => (
  <div className="flex items-center gap-3 pt-2">
    <span className="text-sm font-semibold text-foreground">{label}</span>
    <Separator className="flex-1" />
  </div>
);

interface ModePickerProps {
  selected: "simple" | "flow_builder";
  onSelect: (mode: "simple" | "flow_builder") => void;
}

const modeCardClass = (isSelected: boolean) =>
  [
    "relative flex-1 cursor-pointer rounded-xl border-2 px-5 py-6 transition-all",
    isSelected ? "border-info bg-info/10" : "border-border bg-background",
  ].join(" ");

const modeIconClass = (isSelected: boolean) =>
  [
    "mb-4 flex size-10 items-center justify-center rounded-[10px]",
    isSelected ? "bg-info/15 text-info" : "bg-muted text-muted-foreground",
  ].join(" ");

const ModePicker: React.FC<ModePickerProps> = ({ selected, onSelect }) => (
  <div className="flex gap-4 py-2">
    <div onClick={() => onSelect("simple")} className={modeCardClass(selected === "simple")}>
      <div className={modeIconClass(selected === "simple")}>
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M8 7h8M8 12h8M8 17h5" />
        </svg>
      </div>
      <span className="mb-1 block text-[15px] font-semibold text-foreground">Simple Mode</span>
      <span className="block text-[13px] text-muted-foreground">Pick guardrails from a list. All run in parallel.</span>
    </div>

    <div onClick={() => onSelect("flow_builder")} className={modeCardClass(selected === "flow_builder")}>
      <Badge variant="secondary" className="absolute top-3 right-3 text-[10px] font-semibold">
        NEW
      </Badge>
      <div className={modeIconClass(selected === "flow_builder")}>
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
        </svg>
      </div>
      <span className="mb-1 block text-[15px] font-semibold text-foreground">Flow Builder</span>
      <span className="block text-[13px] text-muted-foreground">Define steps, conditions, and error responses.</span>
    </div>
  </div>
);

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
  const form = useZodForm(policySchema, { defaultValues: EMPTY_VALUES });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [resolvedGuardrails, setResolvedGuardrails] = useState<string[]>([]);
  const [modelConditionType, setModelConditionType] = useState<ModelConditionType>("model");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [step, setStep] = useState<"pick_mode" | "simple_form">("pick_mode");
  const [selectedMode, setSelectedMode] = useState<"simple" | "flow_builder">("simple");
  const { userId, userRole } = useAuthorized();

  // Only consider it "editing" if editingPolicy has a policy_id (real existing policy)
  const isEditing = !!editingPolicy?.policy_id;

  useEffect(() => {
    if (visible && editingPolicy) {
      const modelCondition = editingPolicy.condition?.model;
      const isRegex = modelCondition && /[.*+?^${}()|[\]\\]/.test(modelCondition);
      setModelConditionType(isRegex ? "regex" : "model");

      form.reset(toFormValues(editingPolicy));

      if (editingPolicy.policy_id && accessToken) {
        loadResolvedGuardrails(editingPolicy.policy_id);
      }

      // If editing a pipeline policy, go directly to flow builder
      if (editingPolicy.pipeline) {
        onClose();
        onOpenFlowBuilder();
        return;
      }
      // If editing a simple policy, skip mode picker
      setStep("simple_form");
    } else if (visible) {
      form.reset(EMPTY_VALUES);
      setResolvedGuardrails([]);
      setModelConditionType("model");
      setSelectedMode("simple");
      setStep("pick_mode");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, editingPolicy, form]);

  useEffect(() => {
    if (visible && accessToken) {
      loadAvailableModels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, accessToken]);

  const loadAvailableModels = async () => {
    if (!accessToken) return;
    try {
      const response = await modelAvailableCall(accessToken, userId, userRole);
      if (response?.data) {
        const models = response.data.map((m: any) => m.id || m.model_name).filter(Boolean);
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

  const refreshResolvedGuardrails = (changed: Partial<PolicyFormValues>) => {
    setResolvedGuardrails(resolveGuardrails({ ...form.getValues(), ...changed }, existingPolicies));
  };

  const handleClose = () => {
    form.reset(EMPTY_VALUES);
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

  const handleSubmit = async (values: PolicyFormValues) => {
    try {
      setIsSubmitting(true);

      if (!accessToken) {
        throw new Error("No access token available");
      }

      const data = buildPolicyRequest(values);

      if (isEditing && editingPolicy) {
        await updatePolicy(accessToken, editingPolicy.policy_id, data as PolicyUpdateRequest);
        toast.success("Policy updated successfully");
      } else {
        await createPolicy(accessToken, data as PolicyCreateRequest);
        toast.success("Policy created successfully");
      }

      form.reset(EMPTY_VALUES);
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save policy:", error);
      toast.fromError("Failed to save policy: " + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsSubmitting(false);
    }
  };

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
      <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[620px]">
          <DialogHeader>
            <DialogTitle>Create New Policy</DialogTitle>
          </DialogHeader>
          <ModePicker selected={selectedMode} onSelect={setSelectedMode} />

          {selectedMode === "flow_builder" && (
            <Alert variant="info" className="mt-4 border border-info/20 bg-info/10">
              <AlertTitle>You&apos;ll be taken to the Flow Builder to design your policy logic visually.</AlertTitle>
            </Alert>
          )}

          <div className="mt-6 flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="button" onClick={handleModeConfirm}>
              {selectedMode === "flow_builder" ? "Continue to Builder" : "Create Policy"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // ── Simple Form Step ──────────────────────────────────────────────────────
  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>{isEditing ? "Edit Policy" : "Create New Policy"}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField control={form.control} name="policy_name" label="Policy Name">
                {({ ref, ...control }) => (
                  <Input
                    {...control}
                    ref={ref}
                    placeholder="e.g., global-baseline, healthcare-compliance"
                    disabled={isEditing}
                  />
                )}
              </FormField>

              <FormField control={form.control} name="description" label="Description">
                {({ ref, ...control }) => (
                  <Textarea {...control} ref={ref} rows={2} placeholder="Describe what this policy does..." />
                )}
              </FormField>

              <SectionHeading label="Inheritance" />

              <FormField
                control={form.control}
                name="inherit"
                label={labelWithHint(
                  "Inherit From",
                  "Inherit guardrails from another policy. The child policy will include all guardrails from the parent.",
                )}
              >
                {({ id, value, onChange }) => (
                  <SearchSelect
                    inputId={id}
                    options={policyOptions}
                    value={value}
                    onValueChange={(selected) => {
                      onChange(selected);
                      refreshResolvedGuardrails({ inherit: selected });
                    }}
                    placeholder="Select a parent policy (optional)"
                    className="h-9"
                  />
                )}
              </FormField>

              <SectionHeading label="Guardrails" />

              <FormField
                control={form.control}
                name="guardrails_add"
                label={labelWithHint(
                  "Guardrails to Add",
                  "These guardrails will be added to requests matching this policy",
                )}
              >
                {({ value, onChange }) => (
                  <MultiSelect
                    options={guardrailOptions}
                    value={value}
                    onValueChange={(selected) => {
                      onChange(selected);
                      refreshResolvedGuardrails({ guardrails_add: selected });
                    }}
                    placeholder="Select guardrails to add"
                  />
                )}
              </FormField>

              <FormField
                control={form.control}
                name="guardrails_remove"
                label={labelWithHint(
                  "Guardrails to Remove",
                  "These guardrails will be removed from inherited guardrails",
                )}
              >
                {({ value, onChange }) => (
                  <MultiSelect
                    options={guardrailOptions}
                    value={value}
                    onValueChange={(selected) => {
                      onChange(selected);
                      refreshResolvedGuardrails({ guardrails_remove: selected });
                    }}
                    placeholder="Select guardrails to remove (from inherited)"
                  />
                )}
              </FormField>

              {resolvedGuardrails.length > 0 && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>Resolved Guardrails</AlertTitle>
                  <AlertDescription>
                    <span className="mb-2 block text-muted-foreground">
                      These are the final guardrails that will be applied (including inheritance):
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {resolvedGuardrails.map((g) => (
                        <StatusBadge key={g} tone="info" label={g} />
                      ))}
                    </div>
                  </AlertDescription>
                </Alert>
              )}

              <SectionHeading label="Conditions (Optional)" />

              <Alert variant="info">
                <Info />
                <AlertTitle>Model Scope</AlertTitle>
                <AlertDescription>
                  By default, this policy will run on all models. You can optionally restrict it to specific models
                  below.
                </AlertDescription>
              </Alert>

              <div role="group" className="flex w-full flex-col gap-3">
                <span className="text-sm leading-snug font-medium text-foreground">Model Condition Type</span>
                <RadioGroup
                  value={modelConditionType}
                  onValueChange={(value) => {
                    setModelConditionType(value as ModelConditionType);
                    form.setValue("model_condition", "");
                  }}
                  className="flex flex-row gap-6"
                >
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <RadioGroupItem value="model" />
                    Select Model
                  </label>
                  <label className="flex cursor-pointer items-center gap-2 text-sm">
                    <RadioGroupItem value="regex" />
                    Custom Regex Pattern
                  </label>
                </RadioGroup>
              </div>

              <FormField
                control={form.control}
                name="model_condition"
                label={labelWithHint(
                  modelConditionType === "model" ? "Model (Optional)" : "Regex Pattern (Optional)",
                  modelConditionType === "model"
                    ? "Select a specific model to apply this policy to. Leave empty to apply to all models."
                    : "Enter a regex pattern to match models (e.g., gpt-4.* or bedrock/.*). Leave empty to apply to all models.",
                )}
              >
                {({ ref, id, value, onChange, ...control }) =>
                  modelConditionType === "model" ? (
                    <SearchSelect
                      inputId={id}
                      options={availableModels.map((model) => ({ label: model, value: model }))}
                      value={value}
                      onValueChange={onChange}
                      placeholder="Leave empty to apply to all models"
                      className="h-9"
                    />
                  ) : (
                    <Input
                      {...control}
                      id={id}
                      ref={ref}
                      value={value}
                      onChange={onChange}
                      placeholder="Leave empty to apply to all models (e.g., gpt-4.* or bedrock/claude-.*)"
                    />
                  )
                }
              </FormField>
            </FieldGroup>

            <div className="mt-6 flex justify-end gap-2">
              <Button type="button" variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                type="button"
                onClick={form.handleSubmit(handleSubmit)}
                disabled={isSubmitting}
                aria-busy={isSubmitting}
              >
                {isSubmitting && <UiLoadingSpinner className="size-4" />}
                {isEditing ? "Update Policy" : "Create Policy"}
              </Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddPolicyForm;
