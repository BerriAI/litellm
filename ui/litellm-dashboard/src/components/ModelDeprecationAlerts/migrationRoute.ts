import { parseAsString, parseAsStringEnum } from "nuqs";
import { AlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { uiHref } from "@/utils/uiHref";

export const migrationUrlParsers = {
  model: parseAsString,
  team: parseAsString,
  state: parseAsStringEnum<AlertStatus>(Object.values(AlertStatus)).withDefault(AlertStatus.Open),
};

export function migrationHref(model?: string) {
  const base = `${uiHref("alerts")}?type=model-deprecations`;
  return model ? `${base}&model=${encodeURIComponent(model)}` : base;
}
