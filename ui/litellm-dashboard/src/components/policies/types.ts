export interface Policy {
  policy_id: string;
  policy_name: string;
  inherit: string | null;
  description: string | null;
  guardrails_add: string[];
  guardrails_remove: string[];
  condition: PolicyCondition | null;
  pipeline?: GuardrailPipeline | null;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface PolicyCondition {
  model?: string;
}

export interface PipelineStep {
  guardrail: string;
  on_fail: "block" | "allow" | "next" | "modify_response";
  on_pass: "allow" | "block" | "next" | "modify_response";
  pass_data?: boolean;
  modify_response_message?: string | null;
}

export interface GuardrailPipeline {
  mode: "pre_call" | "post_call";
  steps: PipelineStep[];
}

export interface PolicyAttachment {
  attachment_id: string;
  policy_name: string;
  scope: string | null;
  teams: string[];
  keys: string[];
  models: string[];
  tags: string[];
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface PolicyCreateRequest {
  policy_name: string;
  inherit?: string;
  description?: string;
  guardrails_add?: string[];
  guardrails_remove?: string[];
  condition?: PolicyCondition;
  pipeline?: GuardrailPipeline | null;
}

export interface PolicyUpdateRequest {
  policy_name?: string;
  inherit?: string;
  description?: string;
  guardrails_add?: string[];
  guardrails_remove?: string[];
  condition?: PolicyCondition;
  pipeline?: GuardrailPipeline | null;
}

export interface PolicyAttachmentCreateRequest {
  policy_name: string;
  scope?: string;
  teams?: string[];
  keys?: string[];
  models?: string[];
  tags?: string[];
}

export interface PolicyListResponse {
  policies: Policy[];
  total_count: number;
}

export interface PolicyAttachmentListResponse {
  attachments: PolicyAttachment[];
  total_count: number;
}

export interface PipelineStepResult {
  guardrail_name: string;
  outcome: "pass" | "fail" | "error";
  action_taken: string;
  modified_data: Record<string, any> | null;
  error_detail: string | null;
  duration_seconds: number | null;
}

export interface PipelineTestResult {
  terminal_action: string;
  step_results: PipelineStepResult[];
  modified_data: Record<string, any> | null;
  error_message: string | null;
  modify_response_message: string | null;
}
