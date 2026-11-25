import type { AlertingFormValues } from "./schema";

interface AlertingSettingValue {
  field_name: string;
  field_value: unknown;
}

export const isEmptyAlertingSubmission = (formData: AlertingFormValues) =>
  Object.values(formData).every((value) => {
    if (typeof value === "boolean") {
      return false;
    }
    return value === "" || value === null || value === undefined;
  });

export const toAlertingWrites = (formValues: Record<string, unknown>, settings: AlertingSettingValue[]) => {
  const currentValues = Object.fromEntries(settings.map((setting) => [setting.field_name, setting.field_value]));
  const { slack_alerting: slackAlerting, ...alertingArgs } = { ...formValues, ...currentValues };
  return { alertingArgs, slackAlerting };
};
