import { parseAsString, parseAsStringEnum } from "nuqs";
import { AlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { uiHref } from "@/utils/uiHref";

export const migrationUrlParsers = {
  model: parseAsString,
  team: parseAsString,
  state: parseAsStringEnum<AlertStatus>(Object.values(AlertStatus)).withDefault(AlertStatus.Open),
};

export const migrationUrlOptions = { urlKeys: { model: "deprecation_model", team: "deprecation_team" } };

export function migrationHref(model?: string) {
  const base = `${uiHref("models-and-endpoints")}?tab=model-deprecations`;
  return model ? `${base}&deprecation_model=${encodeURIComponent(model)}` : base;
}
