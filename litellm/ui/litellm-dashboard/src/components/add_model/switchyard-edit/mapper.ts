import { emptySwitchyardConfig, switchyardConfigSchema, switchyardRouterPayload } from "../switchyard_config";
import type { SwitchyardEditValues } from "./schema";

export function switchyardEditDefaults(initial: { name: string; config: unknown }) {
  const parsed = switchyardConfigSchema.safeParse(initial.config);
  return {
    values: { name: initial.name, config: parsed.success ? parsed.data : emptySwitchyardConfig() },
    error: parsed.success ? null : "The stored Switchyard configuration is invalid. Correct it before saving.",
  };
}

export const toSwitchyardEditPayload = ({ name, config }: SwitchyardEditValues) =>
  switchyardRouterPayload(name, config);
