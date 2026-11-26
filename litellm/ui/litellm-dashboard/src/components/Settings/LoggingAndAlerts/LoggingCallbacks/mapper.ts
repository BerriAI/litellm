import type { components } from "@/lib/http/schema";

import type { CallbackFormValues, SlackFormValues } from "./schema";

type ConfigPayload = components["schemas"]["ConfigYAML"];

export const callbackToForm = (
  name: string,
  variables: Record<string, unknown>,
  params: string[],
  config: { dynamic_params?: Record<string, { type?: string; default?: unknown }> } | undefined,
): CallbackFormValues => ({
  ...Object.fromEntries(
    Object.entries(variables).flatMap(([key, value]) => {
      const fieldName = params.find((param) => param.toUpperCase() === key.toUpperCase()) ?? key;
      const param = config?.dynamic_params?.[fieldName];
      if (value == null && (param?.type === "select" || param?.default !== undefined)) {
        return [];
      }
      return [[fieldName, value ?? ""]];
    }),
  ),
  callback: name,
});

export const callbackToPayload = (formValues: CallbackFormValues, callbackName: string): ConfigPayload => ({
  environment_variables: Object.fromEntries(Object.entries(formValues).filter(([, value]) => value !== undefined)),
  litellm_settings: {
    success_callback: [callbackName],
  },
});

type SlackPayload = {
  general_settings: Pick<NonNullable<ConfigPayload["general_settings"]>, "alert_to_webhook_url" | "alert_types">;
};

export const slackToPayload = ({ activeAlerts, ...webhookUrls }: SlackFormValues): SlackPayload => ({
  general_settings: {
    alert_to_webhook_url: webhookUrls,
    alert_types: activeAlerts,
  },
});
