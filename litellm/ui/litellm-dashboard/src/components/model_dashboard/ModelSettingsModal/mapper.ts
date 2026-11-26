import type { ProxyConfigResponse } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import type { ModelSettingsFields } from "./schema";

export const configToModelSettings = (config: ProxyConfigResponse | undefined): ModelSettingsFields => ({
  store_model_in_db: config?.find((field) => field.field_name === "store_model_in_db")?.field_value ?? false,
});
