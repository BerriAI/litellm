import { Dayjs } from "dayjs";
import { ptuPickerToUtcIso } from "./ptuDatetime";
import { PTU_COUNT_FIELD, PTU_RATE_FIELD } from "./ptuValidation";

export const PTU_MODEL_INFO_FIELDS: readonly string[] = [
  PTU_COUNT_FIELD,
  PTU_RATE_FIELD,
  "ptu_effective_from",
  "ptu_effective_to",
];

export interface PtuFormValues {
  ptu_count?: string | number | null;
  cost_per_ptu_per_hour?: string | number | null;
  ptu_effective_from?: Dayjs | null;
  ptu_effective_to?: Dayjs | null;
}

const ptuNumber = (value: string | number | null | undefined): number | null =>
  value !== undefined && value !== null && value !== "" ? Number(value) : null;

/**
 * Fold the PTU form values into the model_info an edit is about to save.
 *
 * When PTU cost attribution is off the four fields are stripped rather than sent as null:
 * the form does not render them, so a null would be an explicit clear of config the operator
 * never saw, and any PTU field present in the payload is rejected by the proxy.
 */
export const applyPtuModelInfo = (
  modelInfo: Record<string, unknown>,
  values: PtuFormValues,
  enabled: boolean,
): Record<string, unknown> => {
  if (!enabled) {
    return Object.fromEntries(Object.entries(modelInfo).filter(([key]) => !PTU_MODEL_INFO_FIELDS.includes(key)));
  }
  return {
    ...modelInfo,
    ptu_count: ptuNumber(values.ptu_count),
    cost_per_ptu_per_hour: ptuNumber(values.cost_per_ptu_per_hour),
    ptu_effective_from: ptuPickerToUtcIso(values.ptu_effective_from),
    ptu_effective_to: ptuPickerToUtcIso(values.ptu_effective_to),
  };
};
