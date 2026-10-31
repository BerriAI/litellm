import { USD, type Currency } from "@/lib/money";
import { useUISettings } from "./useUISettings";

export function useCurrency(): Currency {
  const { data } = useUISettings();
  const currency: unknown = data?.currency;
  if (typeof currency !== "object" || currency === null) return USD;
  if (!("code" in currency) || typeof currency.code !== "string" || !/^[A-Z]{3}$/.test(currency.code)) return USD;
  if (!("usd_rate" in currency) || typeof currency.usd_rate !== "number") return USD;
  if (!Number.isFinite(currency.usd_rate) || currency.usd_rate <= 0) return USD;
  return { code: currency.code, usd_rate: currency.usd_rate };
}
