import { PolicyAttachmentCreateRequest } from "./types";

/**
 * Builds a PolicyAttachmentCreateRequest from form values.
 *
 * @param formValues - The raw form field values (from form.getFieldsValue)
 * @param scopeType - Whether the attachment is "global" or "specific"
 */
export function buildAttachmentData(
  formValues: Record<string, any>,
  scopeType: "global" | "specific"
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
  return data;
}
