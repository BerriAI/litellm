import type { ParseKeys } from "i18next";

import i18n from "@/i18n/i18n";
import type { components } from "@/lib/http/schema";

export type PricingTier = components["schemas"]["PriceTier"];
export type DeploymentPricing = components["schemas"]["DeploymentPricing"];

export const SERVICE_TIERS = ["standard", "flex", "balanced", "priority", "ultrafast", "batches"] as const;
export type ServiceTier = (typeof SERVICE_TIERS)[number];

export const SERVICE_TIER_LABELS: Readonly<Record<ServiceTier, ParseKeys>> = {
  standard: "deploymentPricing.tiers.standard",
  flex: "deploymentPricing.tiers.flex",
  balanced: "deploymentPricing.tiers.balanced",
  priority: "deploymentPricing.tiers.priority",
  ultrafast: "deploymentPricing.tiers.ultrafast",
  batches: "deploymentPricing.tiers.batches",
};

const UNITS = ["token", "second", "image", "pixel", "character", "query", "request"] as const;
type Unit = (typeof UNITS)[number];

const CHARGES = ["input", "cache_read", "explicit_cache_read", "cache_write", "output"] as const;
type Charge = (typeof CHARGES)[number];

const MODALITIES = ["", "text", "image", "audio", "video", "reasoning"] as const;
type Modality = (typeof MODALITIES)[number];

export type CostKey =
  | {
      readonly kind: "price";
      readonly charge: Charge;
      readonly modality: Modality;
      readonly cacheDuration: string;
      readonly unit: Unit;
      readonly threshold: number | null;
      readonly tier: ServiceTier;
    }
  | { readonly kind: "unknown"; readonly key: string };

const TIER_SUFFIX = /^(.+)_(flex|balanced|priority|ultrafast|batches)$/;
const THRESHOLD_SUFFIX = /^(.+)_above_(\d+)k_tokens$/;
const CACHE_DURATION_SUFFIX = /^(.+)_above_(\d+hr)$/;
const DIRECT_PRICE = /^(input|output)_cost_per_(?:(text|image|audio|video|reasoning)_(?:per_)?)?([a-z]+)$/;
const CACHE_PRICE = /^(explicit_)?cache_(read|creation)_input_(?:(image|audio)_)?token_cost$/;

const splitSuffix = (key: string, pattern: RegExp): readonly [string, string | null] => {
  const match = pattern.exec(key);
  return match ? [match[1], match[2]] : [key, null];
};

const isUnit = (value: string): value is Unit => (UNITS as readonly string[]).includes(value);

const parseBase = (base: string): Pick<Extract<CostKey, { kind: "price" }>, "charge" | "modality" | "unit"> | null => {
  const direct = DIRECT_PRICE.exec(base);
  if (direct) {
    const [, charge, modality = "", unit] = direct;
    return isUnit(unit) ? { charge: charge as Charge, modality: modality as Modality, unit } : null;
  }
  const cache = CACHE_PRICE.exec(base);
  if (cache) {
    const [, explicit, kind, modality = ""] = cache;
    if (kind === "creation")
      return explicit ? null : { charge: "cache_write", modality: modality as Modality, unit: "token" };
    return { charge: explicit ? "explicit_cache_read" : "cache_read", modality: modality as Modality, unit: "token" };
  }
  return null;
};

export const parseCostKey = (key: string): CostKey => {
  const [withoutTier, tier] = splitSuffix(key, TIER_SUFFIX);
  const [withoutThreshold, thousands] = splitSuffix(withoutTier, THRESHOLD_SUFFIX);
  const [base, cacheDuration] = splitSuffix(withoutThreshold, CACHE_DURATION_SUFFIX);
  const parsed = parseBase(base);
  if (!parsed) return { kind: "unknown", key };
  return {
    kind: "price",
    ...parsed,
    cacheDuration: cacheDuration ?? "",
    threshold: thousands === null ? null : Number(thousands) * 1000,
    tier: (tier ?? "standard") as ServiceTier,
  };
};

type Price = Extract<CostKey, { kind: "price" }>;

type LabelForm = "start" | "inline";

const CHARGE_LABELS: Readonly<Record<LabelForm, Readonly<Record<Charge, ParseKeys>>>> = {
  start: {
    input: "deploymentPricing.charges.input",
    cache_read: "deploymentPricing.charges.cacheRead",
    explicit_cache_read: "deploymentPricing.charges.explicitCacheRead",
    cache_write: "deploymentPricing.charges.cacheWrite",
    output: "deploymentPricing.charges.output",
  },
  inline: {
    input: "deploymentPricing.chargesInline.input",
    cache_read: "deploymentPricing.chargesInline.cacheRead",
    explicit_cache_read: "deploymentPricing.chargesInline.explicitCacheRead",
    cache_write: "deploymentPricing.chargesInline.cacheWrite",
    output: "deploymentPricing.chargesInline.output",
  },
};

const MODALITY_LABELS: Readonly<Record<LabelForm, Readonly<Record<Exclude<Modality, "">, ParseKeys>>>> = {
  start: {
    text: "deploymentPricing.modalities.text",
    image: "deploymentPricing.modalities.image",
    audio: "deploymentPricing.modalities.audio",
    video: "deploymentPricing.modalities.video",
    reasoning: "deploymentPricing.modalities.reasoning",
  },
  inline: {
    text: "deploymentPricing.modalitiesInline.text",
    image: "deploymentPricing.modalitiesInline.image",
    audio: "deploymentPricing.modalitiesInline.audio",
    video: "deploymentPricing.modalitiesInline.video",
    reasoning: "deploymentPricing.modalitiesInline.reasoning",
  },
};

const UNIT_LABELS: Readonly<Record<Unit, ParseKeys>> = {
  token: "deploymentPricing.units.token",
  second: "deploymentPricing.units.second",
  image: "deploymentPricing.units.image",
  pixel: "deploymentPricing.units.pixel",
  character: "deploymentPricing.units.character",
  query: "deploymentPricing.units.query",
  request: "deploymentPricing.units.request",
};

const chargeName = ({ charge, modality }: Price, form: LabelForm): string =>
  modality
    ? i18n.t("deploymentPricing.withModality", {
        modality: i18n.t(MODALITY_LABELS[form][modality]),
        charge: i18n.t(CHARGE_LABELS.inline[charge]),
      })
    : i18n.t(CHARGE_LABELS[form][charge]);

const chargeLabel = (price: Price, form: LabelForm): string =>
  price.cacheDuration
    ? i18n.t("deploymentPricing.withCacheDuration", { charge: chargeName(price, form), duration: price.cacheDuration })
    : chargeName(price, form);

const unitLabel = (unit: Unit): string => i18n.t(UNIT_LABELS[unit]);

const rowLabel = (price: Price): string =>
  i18n.t("deploymentPricing.rowLabel", { charge: chargeLabel(price, "start"), unit: unitLabel(price.unit) });

const rowRank = (price: Price): readonly number[] => [
  CHARGES.indexOf(price.charge),
  UNITS.indexOf(price.unit),
  MODALITIES.indexOf(price.modality),
  parseInt(price.cacheDuration || "0", 10),
];

const compareRanks = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((result, value, index) => result || value - b[index], 0);

const PRICE_FORMAT_OPTIONS: Intl.NumberFormatOptions = {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 10,
  useGrouping: false,
};

const PRICE_FORMAT = new Intl.NumberFormat("en-US", PRICE_FORMAT_OPTIONS);

export const formatRawPrice = (value: number): string =>
  value === 0 ? i18n.t("deploymentPricing.free") : PRICE_FORMAT.format(value);

const formatUnitPrice = (value: number, unit: Unit): string =>
  formatRawPrice(unit === "token" ? value * 1_000_000 : value);

export interface PriceRow {
  readonly label: string;
  readonly prices: Readonly<Partial<Record<ServiceTier, string>>>;
}

export interface PriceGroup {
  readonly threshold: number | null;
  readonly rows: readonly PriceRow[];
}

export interface UnknownPrice {
  readonly key: string;
  readonly label: string;
  readonly price: string;
}

export interface PricingTable {
  readonly tiers: readonly ServiceTier[];
  readonly groups: readonly PriceGroup[];
  readonly hasBlankCells: boolean;
  readonly unknown: readonly UnknownPrice[];
}

const PER_UNIT_KEY = /^(.*?)_?cost_per_(.+)$/;
const COST_SUFFIX = /^(.+?)_cost(?:_(.+))?$/;

const sentenceCase = (text: string): string => {
  const words = text.replaceAll("_", " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};

export const fallbackCostLabel = (key: string): string => {
  const perUnit = PER_UNIT_KEY.exec(key);
  if (perUnit) {
    return i18n.t("deploymentPricing.rowLabel", {
      charge: perUnit[1] ? sentenceCase(perUnit[1]) : i18n.t("deploymentPricing.fallbackCost"),
      unit: perUnit[2].replaceAll("_", " "),
    });
  }
  const suffixed = COST_SUFFIX.exec(key);
  if (suffixed) return sentenceCase([suffixed[1], suffixed[2]].filter(Boolean).join(" "));
  return sentenceCase(key);
};

interface ParsedEntry {
  readonly key: string;
  readonly value: number;
  readonly cost: CostKey;
}

interface PriceEntry extends ParsedEntry {
  readonly cost: Price;
}

const isPriceEntry = (entry: ParsedEntry): entry is PriceEntry => entry.cost.kind === "price";

const parseCosts = (costs: Readonly<Record<string, number>>): readonly ParsedEntry[] =>
  Object.entries(costs).map(([key, value]) => ({ key, value, cost: parseCostKey(key) }));

const uniqueBy = <T, K>(items: readonly T[], keyOf: (item: T) => K): readonly T[] =>
  items.filter((item, index) => items.findIndex((other) => keyOf(other) === keyOf(item)) === index);

const byRowRank = (a: PriceEntry, b: PriceEntry): number => compareRanks(rowRank(a.cost), rowRank(b.cost));

const groupRows = (entries: readonly PriceEntry[]): readonly PriceRow[] =>
  [...uniqueBy(entries, ({ cost }) => rowLabel(cost))].sort(byRowRank).map(({ cost }) => ({
    label: rowLabel(cost),
    prices: Object.fromEntries(
      entries
        .filter((other) => rowLabel(other.cost) === rowLabel(cost))
        .map((other) => [other.cost.tier, formatUnitPrice(other.value, other.cost.unit)]),
    ),
  }));

export const buildPricingTable = (costs: Readonly<Record<string, number>>): PricingTable => {
  const parsed = parseCosts(costs);
  const prices = parsed.filter(isPriceEntry);
  const thresholds = [...uniqueBy(prices, ({ cost }) => cost.threshold)]
    .map(({ cost }) => cost.threshold)
    .sort((a, b) => (a ?? 0) - (b ?? 0));
  const tiers = SERVICE_TIERS.filter((tier) => prices.some(({ cost }) => cost.tier === tier));
  const groups = thresholds.map((threshold) => ({
    threshold,
    rows: groupRows(prices.filter(({ cost }) => cost.threshold === threshold)),
  }));
  return {
    tiers,
    groups,
    hasBlankCells: groups.some(({ rows }) => rows.some(({ prices: row }) => tiers.some((tier) => !(tier in row)))),
    unknown: parsed
      .filter((entry) => !isPriceEntry(entry))
      .map(({ key, value }) => ({ key, label: fallbackCostLabel(key), price: formatRawPrice(value) })),
  };
};

export const formatTokenCount = (tokens: number): string => {
  if (tokens >= 1_000_000) return `${+(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${+(tokens / 1_000).toFixed(1)}K`;
  return String(tokens);
};

const tierPart = ({ cost, value }: PriceEntry): string => {
  const charge = chargeLabel(cost, "inline");
  const price = formatUnitPrice(value, cost.unit);
  return cost.unit === "token"
    ? i18n.t("deploymentPricing.tierPrice", { price, charge })
    : i18n.t("deploymentPricing.tierPricePerUnit", { price, charge, unit: unitLabel(cost.unit) });
};

export const describePricingTier = ({ range: [low, high], costs }: PricingTier): string => {
  const parsed = parseCosts(costs);
  const parts = [
    ...[...parsed.filter(isPriceEntry)].sort(byRowRank).map(tierPart),
    ...parsed
      .filter((entry) => !isPriceEntry(entry))
      .map(({ key, value }) =>
        i18n.t("deploymentPricing.tierPrice", {
          price: formatRawPrice(value),
          charge: fallbackCostLabel(key).toLowerCase(),
        }),
      ),
  ];
  return i18n.t("deploymentPricing.tierRange", {
    low: formatTokenCount(low),
    high: formatTokenCount(high),
    prices: parts.join(" / "),
  });
};
