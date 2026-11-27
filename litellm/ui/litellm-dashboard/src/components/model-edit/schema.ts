import type { Dayjs } from "dayjs";
import type { TFunction } from "i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
import type { CacheControlInjectionPoint } from "@/components/add_model/cache_control_settings";
import { parseModelExtraParams } from "@/utils/modelEditParams";
import {
  MAX_COST_PER_PTU_PER_HOUR,
  MAX_PTU_COUNT,
  isFilledPtuValue,
  isNonNegativePtuRate,
  isPositiveWholePtuCount,
  ptuWindowIsOrdered,
} from "@/utils/ptuValidation";

const PRICING_FIELDS = [
  "input_cost",
  "output_cost",
  "cache_read_cost",
  "cache_write_cost",
  "cache_write_cost_1h",
] as const;
export type TouchedPricingField = (typeof PRICING_FIELDS)[number];

const scalar = z.union([z.string(), z.number(), z.null()]).optional();
const textish = z.string().optional();

const isBlank = (value: string | number | null | undefined): boolean => value == null || String(value).trim() === "";

export const perMillionToPerToken = scalar
  .refine((perMillion) => isBlank(perMillion) || (Number.isFinite(Number(perMillion)) && Number(perMillion) >= 0), {
    error: () => i18n.t("modelInfoEditForm.validation.priceNonNegative"),
  })
  .transform((perMillion) => (isBlank(perMillion) ? null : Number(perMillion) / 1_000_000));

const modelEditShape = {
  model_name: textish,
  litellm_model_name: textish,
  api_base: textish,
  custom_llm_provider: textish,
  organization: textish,
  tpm: scalar,
  rpm: scalar,
  max_retries: scalar,
  timeout: scalar,
  stream_timeout: scalar,
  input_cost: perMillionToPerToken,
  output_cost: perMillionToPerToken,
  cache_read_cost: perMillionToPerToken,
  cache_write_cost: perMillionToPerToken,
  cache_write_cost_1h: perMillionToPerToken,
  ptu_count: scalar,
  cost_per_ptu_per_hour: scalar,
  ptu_effective_from: z.custom<Dayjs | null>().nullish(),
  ptu_effective_to: z.custom<Dayjs | null>().nullish(),
  cache_control: z.boolean().optional(),
  cache_control_injection_points: z.array(z.custom<CacheControlInjectionPoint>()).optional(),
  model_access_group: z.array(z.string()).optional(),
  guardrails: z.array(z.string()).optional(),
  vector_store_ids: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
  health_check_model: z.string().nullish(),
  litellm_credential_name: z.string().nullish(),
  litellm_extra_params: textish,
  model_info: textish,
  team_id: textish,
};

export type ModelEditFieldName = keyof typeof modelEditShape;

export const buildModelEditSchema = (
  ptuEnabled: boolean,
  isFieldTouched: (field: TouchedPricingField) => boolean,
  t: TFunction = i18n.t.bind(i18n),
) =>
  z.object(modelEditShape).superRefine((values, ctx) => {
    const reject = (path: ModelEditFieldName, message: string) =>
      ctx.addIssue({ code: "custom", path: [path], message });

    if (parseModelExtraParams(values.litellm_extra_params) === null) {
      reject("litellm_extra_params", t("modelInfoEditForm.validation.invalidJson"));
    }

    // antd validates only mounted fields, and the PTU block does not render when the flag is off.
    if (!ptuEnabled) {
      return;
    }

    if (!isPositiveWholePtuCount(values.ptu_count)) {
      reject("ptu_count", t("modelInfoEditForm.validation.ptuCountRange", { max: MAX_PTU_COUNT.toLocaleString() }));
    }
    if (!isNonNegativePtuRate(values.cost_per_ptu_per_hour)) {
      reject(
        "cost_per_ptu_per_hour",
        t("modelInfoEditForm.validation.ptuRateRange", { max: MAX_COST_PER_PTU_PER_HOUR.toLocaleString() }),
      );
    }
    if (isFilledPtuValue(values.ptu_count) !== isFilledPtuValue(values.cost_per_ptu_per_hour)) {
      const message = t("modelInfoEditForm.validation.ptuPair");
      reject("ptu_count", message);
      reject("cost_per_ptu_per_hour", message);
    }
    if (isFilledPtuValue(values.ptu_count) && !isFilledPtuValue(values.ptu_effective_from)) {
      reject("ptu_effective_from", t("modelInfoEditForm.validation.ptuStartRequired"));
    }
    if (!ptuWindowIsOrdered(values.ptu_effective_from, values.ptu_effective_to)) {
      const message = t("modelInfoEditForm.validation.ptuWindowOrder");
      reject("ptu_effective_from", message);
      reject("ptu_effective_to", message);
    }

    for (const field of PRICING_FIELDS) {
      const value = values[field];
      if (
        isFieldTouched(field) &&
        isFilledPtuValue(values.ptu_count) &&
        isFilledPtuValue(value) &&
        Number(value) !== 0
      ) {
        reject(field, t("modelInfoEditForm.validation.ptuPricing"));
      }
    }
  });

type ModelEditSchema = ReturnType<typeof buildModelEditSchema>;
export type ModelEditFormValues = z.input<ModelEditSchema>;
export type ModelEditSubmitValues = z.output<ModelEditSchema>;
