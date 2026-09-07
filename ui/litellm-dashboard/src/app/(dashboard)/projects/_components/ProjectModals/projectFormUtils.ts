import { ProjectFormValues } from "./ProjectBaseForm";

type ModelLimitEntry = NonNullable<ProjectFormValues["modelLimits"]>[number];

const buildModelLimitMap = (
  entries: readonly ModelLimitEntry[],
  getValue: (entry: ModelLimitEntry) => number | undefined,
): Record<string, number> =>
  Object.fromEntries(
    entries.flatMap((entry) => {
      const value = getValue(entry);
      return entry.model && value != null ? [[entry.model, value] as const] : [];
    }),
  );

const buildMetadata = (entries: ProjectFormValues["metadata"]): Record<string, string> | undefined =>
  entries && Object.fromEntries(entries.flatMap((entry) => (entry.key ? [[entry.key, entry.value] as const] : [])));

const buildProjectApiParams = (values: ProjectFormValues, sendEmpty: boolean) => {
  const limitEntries = values.modelLimits ?? [];
  const modelRpmLimit = buildModelLimitMap(limitEntries, (entry) => entry.rpm);
  const modelTpmLimit = buildModelLimitMap(limitEntries, (entry) => entry.tpm);
  const modelItpmLimit = buildModelLimitMap(limitEntries, (entry) => entry.itpm);
  const modelOtpmLimit = buildModelLimitMap(limitEntries, (entry) => entry.otpm);
  const metadata = buildMetadata(values.metadata);
  const keepLimits = sendEmpty && values.modelLimits !== undefined;
  const keep = (map: Record<string, number>) => keepLimits || Object.keys(map).length > 0;
  const guardrailsParam =
    values.guardrails !== undefined && (sendEmpty || values.guardrails.length > 0)
      ? { guardrails: values.guardrails }
      : {};
  const metadataParam = metadata !== undefined && (sendEmpty || Object.keys(metadata).length > 0) ? { metadata } : {};

  return {
    project_alias: values.project_alias,
    description: values.description,
    models: values.models ?? [],
    max_budget: values.max_budget === undefined ? undefined : Math.round(values.max_budget * 100) / 100,
    blocked: values.isBlocked ?? false,
    ...guardrailsParam,
    ...(keep(modelRpmLimit) && { model_rpm_limit: modelRpmLimit }),
    ...(keep(modelTpmLimit) && { model_tpm_limit: modelTpmLimit }),
    ...(keep(modelItpmLimit) && { model_itpm_limit: modelItpmLimit }),
    ...(keep(modelOtpmLimit) && { model_otpm_limit: modelOtpmLimit }),
    ...metadataParam,
  };
};

/** A new project has nothing to clear, so anything the operator left blank is left out. */
export const buildProjectCreateParams = (values: ProjectFormValues) => buildProjectApiParams(values, false);

/**
 * /project/update leaves an omitted key untouched, so a limit the operator cleared has to go out as
 * an explicitly empty map. Omitting it is what silently kept a removed quota enforced.
 */
export const buildProjectUpdateParams = (values: ProjectFormValues) => buildProjectApiParams(values, true);
