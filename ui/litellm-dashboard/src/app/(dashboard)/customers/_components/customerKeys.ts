import { useQuery } from "@tanstack/react-query";
import { keyListCall, keyUpdateCall, uiSpendLogsCall } from "@/components/networking";
import { monthRange } from "./customerPortalData";

export interface PortalKey {
  token: string;
  alias: string;
  models: string[];
  tpmLimit: number | null;
  rpmLimit: number | null;
  metadata: Record<string, unknown>;
}

const limitMap = (metadata: Record<string, unknown>, field: string): Record<string, number> => {
  const m = metadata[field];
  return m && typeof m === "object" ? (m as Record<string, number>) : {};
};

export const customerLimitsOnKey = (key: PortalKey, customerId: string): { tpm: number | null; rpm: number | null } => ({
  tpm: limitMap(key.metadata, "customer_tpm_limit")[customerId] ?? null,
  rpm: limitMap(key.metadata, "customer_rpm_limit")[customerId] ?? null,
});

export const useAllKeys = (accessToken: string | null) =>
  useQuery({
    queryKey: ["customer-portal", "keys"],
    enabled: Boolean(accessToken),
    queryFn: async (): Promise<PortalKey[]> => {
      const res = (await keyListCall(accessToken!, null, null, null, null, null, 1, 100)) as { keys?: unknown[] };
      return (res.keys ?? [])
        .filter((k): k is Record<string, unknown> => typeof k === "object" && k !== null)
        .map((k) => ({
          token: String(k.token ?? ""),
          alias: String(k.key_alias ?? k.token ?? ""),
          models: Array.isArray(k.models) ? (k.models as string[]) : [],
          tpmLimit: typeof k.tpm_limit === "number" ? k.tpm_limit : null,
          rpmLimit: typeof k.rpm_limit === "number" ? k.rpm_limit : null,
          metadata: (k.metadata && typeof k.metadata === "object" ? k.metadata : {}) as Record<string, unknown>,
        }));
    },
  });

export const useCustomerTrafficKeys = (accessToken: string | null, customerId: string) =>
  useQuery({
    queryKey: ["customer-portal", "traffic-keys", customerId],
    enabled: Boolean(accessToken) && Boolean(customerId),
    queryFn: async (): Promise<Set<string>> => {
      const { start, end } = monthRange(0);
      const fmt = (d: Date) => d.toISOString().slice(0, 19).replace("T", " ");
      const res = (await uiSpendLogsCall({ accessToken: accessToken!, start_date: fmt(start), end_date: fmt(end), page: 1, page_size: 500, params: { end_user: customerId } })) as { data?: { api_key?: string }[] };
      return new Set((res.data ?? []).map((r) => r.api_key ?? "").filter(Boolean));
    },
  });

export const writeCustomerLimits = async (
  accessToken: string,
  key: PortalKey,
  customerId: string,
  limits: { tpm: number | null; rpm: number | null },
) => {
  const tpmMap = { ...limitMap(key.metadata, "customer_tpm_limit") };
  const rpmMap = { ...limitMap(key.metadata, "customer_rpm_limit") };
  if (limits.tpm === null) delete tpmMap[customerId]; else tpmMap[customerId] = limits.tpm;
  if (limits.rpm === null) delete rpmMap[customerId]; else rpmMap[customerId] = limits.rpm;
  await keyUpdateCall(accessToken, { key: key.token, metadata: { ...key.metadata, customer_tpm_limit: tpmMap, customer_rpm_limit: rpmMap } });
};
