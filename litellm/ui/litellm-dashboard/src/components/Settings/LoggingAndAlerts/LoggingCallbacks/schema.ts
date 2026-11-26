import { z } from "zod";

import type { components } from "@/lib/http/schema";

export const callbackSchema = (requiredFields: Readonly<Record<string, string>>) =>
  z.record(z.string(), z.unknown()).superRefine((values, ctx) => {
    for (const [field, message] of Object.entries(requiredFields)) {
      const value = values[field];
      const isEmpty = value == null || value === "" || value === false;
      const isEmptyArray = Array.isArray(value) && value.length === 0;
      if (isEmpty || isEmptyArray) {
        ctx.addIssue({ code: "custom", path: [field], message });
      }
    }
  });

export type CallbackFormValues = z.input<ReturnType<typeof callbackSchema>>;

export type AlertType = components["schemas"]["AlertType"];

export const slackSchema = z.looseObject({ activeAlerts: z.custom<AlertType[]>() });

export type SlackFormValues = z.output<typeof slackSchema>;
