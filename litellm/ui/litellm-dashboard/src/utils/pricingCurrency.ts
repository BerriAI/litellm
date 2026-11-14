export const PRICING_CURRENCIES = ["USD", "EUR", "HUF"] as const;
export type PricingCurrency = (typeof PRICING_CURRENCIES)[number];

export const PRICING_CURRENCY_FIELD = "pricing_currency";
export const USD_PER_CURRENCY_UNIT_FIELD = "usd_per_currency_unit";

const DERIVED_MODEL_INFO_KEYS = new Set(["pricing_source", "usd_pricing"]);

export type PriceUnit =
  | "million_tokens"
  | "million_characters"
  | "second"
  | "image"
  | "query"
  | "page"
  | "pixel"
  | "ptu_hour"
  | "clip"
  | "unit";

const THRESHOLD_SUFFIX = /_above_.*$/;

const UNIT_PATTERNS: ReadonlyArray<readonly [RegExp, PriceUnit]> = [
  [/_per_ptu_per_hour$/, "ptu_hour"],
  [/_per_video_clip$/, "clip"],
  [/_per_second(_|$)/, "second"],
  [/_token(_|$)/, "million_tokens"],
  [/_per_character(_|$)/, "million_characters"],
  [/_per_image(_|$)/, "image"],
  [/_per_query(_|$)/, "query"],
  [/_per_page(_|$)/, "page"],
  [/_per_pixel(_|$)/, "pixel"],
];

const PER_MILLION_UNITS: ReadonlySet<PriceUnit> = new Set(["million_tokens", "million_characters"]);

const segmentUnit = (segment: string): PriceUnit => {
  const unitPart = segment.replace(THRESHOLD_SUFFIX, "");
  return UNIT_PATTERNS.find(([pattern]) => pattern.test(unitPart))?.[1] ?? "unit";
};

export const priceUnit = (path: string): PriceUnit =>
  path
    .split(/[.[\]]+/)
    .filter((segment) => segment !== "")
    .reverse()
    .map(segmentUnit)
    .find((unit) => unit !== "unit") ?? "unit";

export const displayedAmount = (field: string, ratePerUnit: number): number =>
  PER_MILLION_UNITS.has(priceUnit(field)) ? ratePerUnit * 1_000_000 : ratePerUnit;

export const isPricingCurrency = (value: unknown): value is PricingCurrency =>
  typeof value === "string" && (PRICING_CURRENCIES as readonly string[]).includes(value);

export interface CurrencySetting {
  currency: PricingCurrency;
  rate: string;
}

export const storedCurrencySetting = (
  litellmParams: Record<string, unknown> | null | undefined,
  modelInfo?: Record<string, unknown> | null,
): CurrencySetting => {
  const currency = litellmParams?.[PRICING_CURRENCY_FIELD] ?? modelInfo?.[PRICING_CURRENCY_FIELD];
  const rate = litellmParams?.[USD_PER_CURRENCY_UNIT_FIELD] ?? modelInfo?.[USD_PER_CURRENCY_UNIT_FIELD];
  return {
    currency: isPricingCurrency(currency) ? currency : "USD",
    rate: rate == null ? "" : String(rate),
  };
};

export const parsedRate = (setting: CurrencySetting): number | null => {
  if (setting.currency === "USD") return 1;
  const trimmed = setting.rate.trim();
  if (trimmed === "" || !/^\d*\.?\d+(e[-+]?\d+)?$/i.test(trimmed)) return null;
  const rate = Number(trimmed);
  return Number.isFinite(rate) && rate > 0 ? rate : null;
};

export type RateProblem = "required" | "positive";

export const rateProblem = (setting: CurrencySetting): RateProblem | null => {
  if (setting.currency === "USD") return null;
  if (setting.rate.trim() === "") return "required";
  return parsedRate(setting) === null ? "positive" : null;
};

export const currencyPatch = (stored: CurrencySetting, edited: CurrencySetting): Record<string, unknown> => {
  if (edited.currency === "USD") {
    return stored.currency === "USD" ? {} : { [PRICING_CURRENCY_FIELD]: "USD", [USD_PER_CURRENCY_UNIT_FIELD]: null };
  }
  const rateChanged = edited.rate.trim() !== stored.rate.trim() || stored.currency === "USD";
  return {
    ...(edited.currency !== stored.currency ? { [PRICING_CURRENCY_FIELD]: edited.currency } : {}),
    ...(rateChanged || edited.currency !== stored.currency
      ? { [USD_PER_CURRENCY_UNIT_FIELD]: edited.rate.trim() }
      : {}),
  };
};

const isPriceValue = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

const SETTING_KEYS = new Set(["range", "hours_utc", "windows", "weekday_timezone"]);

const flattenPrices = (path: string, value: unknown): ReadonlyArray<readonly [string, number]> => {
  if (isPriceValue(value)) return [[path, value]];
  if (Array.isArray(value)) return value.flatMap((item, index) => flattenPrices(`${path}[${index}]`, item));
  if (value !== null && typeof value === "object") {
    return Object.entries(value).flatMap(([key, item]) =>
      SETTING_KEYS.has(key) ? [] : flattenPrices(`${path}.${key}`, item),
    );
  }
  return [];
};

export const configuredPrices = (model: {
  litellm_params?: Record<string, unknown> | null;
  model_info?: Record<string, unknown> | null;
}): Record<string, number> => {
  const modelInfo = model.model_info ?? {};
  const litellmParams = model.litellm_params ?? {};
  const usdPricing = modelInfo.usd_pricing;
  const overrides = modelInfo.pricing_overrides;
  const fields = new Set<string>([
    ...(usdPricing !== null && typeof usdPricing === "object" ? Object.keys(usdPricing) : []),
    ...(Array.isArray(overrides) ? overrides.filter((field): field is string => typeof field === "string") : []),
  ]);
  return Object.fromEntries(
    [...fields].flatMap((field) => flattenPrices(field, litellmParams[field] ?? modelInfo[field])),
  );
};

export interface PricePreviewRow {
  field: string;
  unit: PriceUnit;
  source: number;
  usd: number | null;
  edited: boolean;
}

export const pricePreview = (
  stored: Record<string, number>,
  edits: Record<string, number | null>,
  setting: CurrencySetting,
): PricePreviewRow[] => {
  const rate = parsedRate(setting);
  const merged = { ...stored, ...edits };
  return Object.entries(merged)
    .flatMap(([field, value]) => (value == null ? [] : [[field, value] as const]))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([field, value]) => ({
      field,
      unit: priceUnit(field),
      source: displayedAmount(field, value),
      usd: rate === null ? null : displayedAmount(field, value * rate),
      edited: field in edits,
    }));
};

export const reinterpretedFields = (rows: PricePreviewRow[], stored: CurrencySetting, edited: CurrencySetting) =>
  stored.currency === edited.currency && stored.rate.trim() === edited.rate.trim()
    ? []
    : rows.filter((row) => !row.edited).map((row) => row.field);

export const usdRate = (
  modelInfo: Record<string, unknown> | null | undefined,
  field: string,
  fallback: unknown,
): unknown => {
  const usdPricing = modelInfo?.usd_pricing;
  if (usdPricing !== null && typeof usdPricing === "object" && field in usdPricing) {
    return (usdPricing as Record<string, unknown>)[field];
  }
  return fallback;
};

export const withoutDerivedPricing = (modelInfo: Record<string, unknown> | null | undefined): Record<string, unknown> =>
  Object.fromEntries(Object.entries(modelInfo ?? {}).filter(([key]) => !DERIVED_MODEL_INFO_KEYS.has(key)));

export const formatAmount = (amount: number): string =>
  amount === 0 ? "0" : Math.abs(amount) >= 0.01 ? amount.toFixed(4).replace(/\.?0+$/, "") : amount.toPrecision(4);
