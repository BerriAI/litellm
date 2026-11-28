import { emptySwitchyardConfig, switchyardConfigSchema, switchyardRouterPayload } from "../switchyard_config";
import type { SwitchyardEditValues } from "./schema";

export function switchyardEditDefaults(initial: { name: string; config: unknown }) {
  const parsed = switchyardConfigSchema.safeParse(initial.config);
  return {
    values: { name: initial.name, config: parsed.success ? parsed.data : emptySwitchyardConfig() },
    valid: parsed.success,
  };
}

export const toSwitchyardEditPayload = ({ name, config }: SwitchyardEditValues) =>
  switchyardRouterPayload(name, config);
