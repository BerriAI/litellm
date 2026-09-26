import { z } from "zod/v4";

import { DEFAULT_POOL_NAME } from "@/components/fairness/workloadClass";
import type { components } from "@/lib/http/schema";

export type FairnessSettings = components["schemas"]["FairnessSettings"];
export type FairnessSettingsResponse = components["schemas"]["FairnessSettingsResponse"];
export type FairnessStatusResponse = components["schemas"]["FairnessStatusResponse"];
export type ModelFairnessStatus = components["schemas"]["ModelFairnessStatus"];
export type WorkloadClassStatus = components["schemas"]["WorkloadClassStatus"];

export { DEFAULT_POOL_NAME };
export const MAX_QUEUE_WAIT_SECONDS = 600;

const numberString = (
  min: number,
  max: number,
  message: string,
  options?: { integer?: boolean; exclusiveMin?: boolean },
) =>
  z.string().refine((value) => {
    const parsed = Number(value);
    if (value.trim() === "" || !Number.isFinite(parsed)) return false;
    if (options?.integer === true && !Number.isInteger(parsed)) return false;
    if (options?.exclusiveMin === true ? parsed <= min : parsed < min) return false;
    return parsed <= max;
  }, message);

const percentString = numberString(0, 100, "Enter a percentage between 0 and 100");
const queueWaitString = numberString(0, MAX_QUEUE_WAIT_SECONDS, `Enter 0 to ${MAX_QUEUE_WAIT_SECONDS} seconds`);

const workloadClassShape = {
  name: z
    .string()
    .trim()
    .min(1, "Enter a class name")
    .max(64, "Keep the name under 64 characters")
    .regex(/^[A-Za-z0-9_.-]+$/, "Use letters, numbers, '_', '.' or '-'")
    .refine((name) => name !== DEFAULT_POOL_NAME, `'${DEFAULT_POOL_NAME}' is reserved for unclassified keys and teams`),
  reserved_share: percentString,
  max_queue_wait_seconds: queueWaitString,
  description: z.string().trim().max(256, "Keep the description under 256 characters"),
};

const workloadClassSchema = z.object(workloadClassShape);

export type WorkloadClassFormValues = z.input<typeof workloadClassSchema>;

export const EMPTY_WORKLOAD_CLASS: WorkloadClassFormValues = {
  name: "",
  reserved_share: "0",
  max_queue_wait_seconds: "0",
  description: "",
};

const fairnessSettingsShape = {
  enabled: z.boolean(),
  workload_classes: z.array(workloadClassSchema),
  default_reserved_share: percentString,
  default_max_queue_wait_seconds: queueWaitString,
  saturation_threshold: percentString,
  saturation_check_cache_ttl: numberString(0, 3600, "Enter 0 to 3600 seconds", { integer: true }),
  max_queue_depth_per_class: numberString(1, 100_000, "Enter 1 to 100000 requests", { integer: true }),
  queue_poll_interval_seconds: numberString(0, 5, "Enter more than 0 and at most 5 seconds", { exclusiveMin: true }),
};

export const fairnessSettingsSchema = z.object(fairnessSettingsShape).superRefine((values, ctx) => {
  values.workload_classes.forEach((workloadClass, index) => {
    const firstIndex = values.workload_classes.findIndex((other) => other.name === workloadClass.name);
    if (workloadClass.name !== "" && firstIndex < index) {
      ctx.addIssue({
        code: "custom",
        message: "This class name is already used",
        path: ["workload_classes", index, "name"],
      });
    }
  });
  const totalShare =
    values.workload_classes.reduce((sum, workloadClass) => sum + Number(workloadClass.reserved_share), 0) +
    Number(values.default_reserved_share);
  if (totalShare > 100) {
    ctx.addIssue({
      code: "custom",
      message: `Reserved shares add up to ${Math.round(totalShare)}%. Keep the total at or below 100%`,
      path: ["default_reserved_share"],
    });
  }
});

export type FairnessSettingsFormValues = z.input<typeof fairnessSettingsSchema>;
export type FairnessSettingsSubmitValues = z.output<typeof fairnessSettingsSchema>;

const fractionToPercent = (fraction: number): string => String(Math.round(fraction * 10_000) / 100);
const percentToFraction = (percent: string): number => Number(percent) / 100;

export const settingsToForm = (settings: FairnessSettings): FairnessSettingsFormValues => ({
  enabled: settings.enabled,
  workload_classes: settings.workload_classes.map((workloadClass) => ({
    name: workloadClass.name,
    reserved_share: fractionToPercent(workloadClass.reserved_share),
    max_queue_wait_seconds: String(workloadClass.max_queue_wait_seconds),
    description: workloadClass.description ?? "",
  })),
  default_reserved_share: fractionToPercent(settings.default_reserved_share),
  default_max_queue_wait_seconds: String(settings.default_max_queue_wait_seconds),
  saturation_threshold: fractionToPercent(settings.saturation_threshold),
  saturation_check_cache_ttl: String(settings.saturation_check_cache_ttl),
  max_queue_depth_per_class: String(settings.max_queue_depth_per_class),
  queue_poll_interval_seconds: String(settings.queue_poll_interval_seconds),
});

export const buildBody = (values: FairnessSettingsSubmitValues): FairnessSettings => ({
  enabled: values.enabled,
  workload_classes: values.workload_classes.map((workloadClass) => ({
    name: workloadClass.name,
    reserved_share: percentToFraction(workloadClass.reserved_share),
    max_queue_wait_seconds: Number(workloadClass.max_queue_wait_seconds),
    description: workloadClass.description === "" ? null : workloadClass.description,
  })),
  default_reserved_share: percentToFraction(values.default_reserved_share),
  default_max_queue_wait_seconds: Number(values.default_max_queue_wait_seconds),
  saturation_threshold: percentToFraction(values.saturation_threshold),
  saturation_check_cache_ttl: Number(values.saturation_check_cache_ttl),
  max_queue_depth_per_class: Number(values.max_queue_depth_per_class),
  queue_poll_interval_seconds: Number(values.queue_poll_interval_seconds),
});
