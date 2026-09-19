import { PolicyAttachmentCreateRequest } from "@/components/policies/types";

export interface AttachmentFormInput {
  policy_name: string;
  teams?: string[];
  keys?: string[];
  models?: string[];
  tags?: string[];
  priority?: number | null;
}

export function buildAttachmentData(
  formValues: AttachmentFormInput,
  scopeType: "global" | "specific",
): PolicyAttachmentCreateRequest {
  const data: PolicyAttachmentCreateRequest = {
    policy_name: formValues.policy_name,
  };
  if (scopeType === "global") {
    data.scope = "*";
  } else {
    if (formValues.teams && formValues.teams.length > 0) data.teams = formValues.teams;
    if (formValues.keys && formValues.keys.length > 0) data.keys = formValues.keys;
    if (formValues.models && formValues.models.length > 0) data.models = formValues.models;
    if (formValues.tags && formValues.tags.length > 0) data.tags = formValues.tags;
  }
  if (typeof formValues.priority === "number") data.priority = formValues.priority;
  return data;
}
