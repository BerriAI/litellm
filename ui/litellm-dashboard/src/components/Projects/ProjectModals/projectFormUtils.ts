import { ProjectFormValues } from "./ProjectBaseForm";

/**
 * Transforms ProjectFormValues into the flat API param shape
 * shared by both create and update endpoints.
 */
export function buildProjectApiParams(values: ProjectFormValues) {
  const modelRpmLimit: Record<string, number> = {};
  const modelTpmLimit: Record<string, number> = {};
  for (const entry of values.modelLimits ?? []) {
    if (entry.model) {
      if (entry.rpm != null) modelRpmLimit[entry.model] = entry.rpm;
      if (entry.tpm != null) modelTpmLimit[entry.model] = entry.tpm;
    }
  }

  const metadata: Record<string, unknown> = {};
  for (const entry of values.metadata ?? []) {
    if (entry.key) metadata[entry.key] = entry.value;
  }

  return {
    project_alias: values.project_alias,
    description: values.description,
    models: values.models ?? [],
    max_budget: values.max_budget,
    blocked: values.isBlocked ?? false,
    ...(Object.keys(modelRpmLimit).length > 0 && {
      model_rpm_limit: modelRpmLimit,
    }),
    ...(Object.keys(modelTpmLimit).length > 0 && {
      model_tpm_limit: modelTpmLimit,
    }),
    ...(Object.keys(metadata).length > 0 && { metadata }),
  };
}
