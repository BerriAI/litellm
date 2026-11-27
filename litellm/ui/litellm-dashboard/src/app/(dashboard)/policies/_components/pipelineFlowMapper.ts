import type {
  GuardrailPipeline,
  PipelineStep,
  Policy,
  PolicyCreateRequest,
  PolicyUpdateRequest,
} from "@/components/policies/types";
import type { PipelineFlowFormValues } from "./pipelineFlowSchema";

export function createDefaultStep(): PipelineStep {
  return {
    guardrail: "",
    on_pass: "next",
    on_fail: "block",
    pass_data: false,
    modify_response_message: null,
  };
}

function derivePipelineFromPolicy(policy: Policy | null | undefined): GuardrailPipeline {
  if (!policy) {
    return { mode: "pre_call", steps: [createDefaultStep()] };
  }
  if (policy.pipeline?.steps?.length) {
    return policy.pipeline;
  }
  const guardrails = policy.guardrails_add || [];
  if (guardrails.length > 0) {
    return {
      mode: policy.pipeline?.mode ?? "pre_call",
      steps: guardrails.map((g) => ({
        guardrail: g,
        on_pass: "next" as const,
        on_fail: "block" as const,
        pass_data: false,
        modify_response_message: null,
      })),
    };
  }
  return { mode: "pre_call", steps: [createDefaultStep()] };
}

export function policyToFlowFormValues(editingPolicy?: Policy | null): PipelineFlowFormValues {
  return {
    policy_name: editingPolicy?.policy_name || "",
    description: editingPolicy?.description || "",
    pipeline: derivePipelineFromPolicy(editingPolicy),
  };
}

export function buildPolicyFlowPayload(values: PipelineFlowFormValues): PolicyCreateRequest | PolicyUpdateRequest {
  const guardrailsFromPipeline = values.pipeline.steps.map((s) => s.guardrail).filter(Boolean);
  return {
    policy_name: values.policy_name,
    description: values.description || undefined,
    guardrails_add: guardrailsFromPipeline,
    guardrails_remove: [],
    pipeline: values.pipeline,
  };
}
