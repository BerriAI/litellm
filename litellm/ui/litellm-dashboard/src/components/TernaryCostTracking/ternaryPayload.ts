import { z } from "zod";

import type {
  TernarySettingsInput,
  TernarySettingsUpdate,
  TernarySettingsView,
} from "@/app/(dashboard)/hooks/ternary/useTernary";
import i18n from "@/i18n/i18n";

export const TERNARY_DEFAULT_BASE_URL = "https://api.ternary.app";

export const FREQUENCY_ITEMS = [
  { value: "daily", labelKey: "loggingAndAlerts.ternary.form.frequencyDaily" },
  { value: "interval", labelKey: "loggingAndAlerts.ternary.form.frequencyInterval" },
] as const;

const intervalSecondsField = z.string().refine((value) => value === "" || /^[1-9]\d*$/.test(value), {
  error: () => i18n.t("loggingAndAlerts.ternary.form.intervalSecondsInvalid"),
});

const baseShape = {
  api_key: z.string(),
  connection_id: z
    .string()
    .trim()
    .min(1, { error: () => i18n.t("loggingAndAlerts.ternary.form.connectionIdRequired") }),
  ternary_base_url: z
    .string()
    .trim()
    .min(1, { error: () => i18n.t("loggingAndAlerts.ternary.form.baseUrlRequired") }),
  frequency: z.enum(["daily", "interval"]),
  interval_seconds: intervalSecondsField,
};

const baseSchema = z.object(baseShape);

const requireIntervalSeconds = (values: z.infer<typeof baseSchema>, ctx: z.RefinementCtx) => {
  if (values.frequency === "interval" && values.interval_seconds === "") {
    ctx.addIssue({
      code: "custom",
      path: ["interval_seconds"],
      message: i18n.t("loggingAndAlerts.ternary.form.intervalSecondsRequired"),
    });
  }
};

export const createSchema = baseSchema
  .extend({
    api_key: z
      .string()
      .trim()
      .min(1, { error: () => i18n.t("loggingAndAlerts.ternary.form.apiKeyRequired") }),
  })
  .superRefine(requireIntervalSeconds);

export const updateSchemaFor = (currentBaseUrl: string) =>
  baseSchema.superRefine((values, ctx) => {
    requireIntervalSeconds(values, ctx);
    if (values.ternary_base_url.trim() !== currentBaseUrl && values.api_key.trim() === "") {
      ctx.addIssue({
        code: "custom",
        path: ["api_key"],
        message: i18n.t("loggingAndAlerts.ternary.form.apiKeyRequiredForBaseUrl"),
      });
    }
  });

export type TernaryFormValues = z.input<typeof baseSchema>;

export const EMPTY_TERNARY_FORM_VALUES: TernaryFormValues = {
  api_key: "",
  connection_id: "",
  ternary_base_url: TERNARY_DEFAULT_BASE_URL,
  frequency: "daily",
  interval_seconds: "",
};

export const formValuesFromSettings = (settings: TernarySettingsView): TernaryFormValues => ({
  api_key: "",
  connection_id: settings.connection_id ?? "",
  ternary_base_url: settings.ternary_base_url ?? TERNARY_DEFAULT_BASE_URL,
  frequency: settings.frequency ?? "daily",
  interval_seconds: settings.interval_seconds ? String(settings.interval_seconds) : "",
});

const intervalSeconds = (values: TernaryFormValues): number | null =>
  values.frequency === "interval" ? Number(values.interval_seconds) : null;

export const buildCreatePayload = (values: TernaryFormValues): TernarySettingsInput => ({
  api_key: values.api_key.trim(),
  connection_id: values.connection_id.trim(),
  ternary_base_url: values.ternary_base_url.trim(),
  frequency: values.frequency,
  interval_seconds: intervalSeconds(values),
});

export const buildUpdatePayload = (values: TernaryFormValues): TernarySettingsUpdate => ({
  connection_id: values.connection_id.trim(),
  ternary_base_url: values.ternary_base_url.trim(),
  frequency: values.frequency,
  interval_seconds: intervalSeconds(values),
  ...(values.api_key.trim() && { api_key: values.api_key.trim() }),
});
