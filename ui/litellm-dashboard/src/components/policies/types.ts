export interface Policy {
  policy_id: string;
  policy_name: string;
  inherit: string | null;
  description: string | null;
  guardrails_add: string[];
  guardrails_remove: string[];
  condition: PolicyCondition | null;
  created_at?: string;
  updated_at?: string;
  created_by?: string;
  updated_by?: string;
}

export interface PolicyCondition {
  model?: string;
}

export interface PolicyAttachment {
  attachment_id: string;
  policy_name: string;
  scope: string | null;
  teams: string[];
  keys: string[];
  models: string[];
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
}

export interface PolicyUpdateRequest {
  policy_name?: string;
  inherit?: string;
  description?: string;
  guardrails_add?: string[];
  guardrails_remove?: string[];
  condition?: PolicyCondition;
}

export interface PolicyAttachmentCreateRequest {
  policy_name: string;
  scope?: string;
  teams?: string[];
  keys?: string[];
  models?: string[];
}

export interface PolicyListResponse {
  policies: Policy[];
  total_count: number;
}

export interface PolicyAttachmentListResponse {
  attachments: PolicyAttachment[];
  total_count: number;
}
