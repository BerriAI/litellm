import { describe, expect, it } from "vitest";

import { buildPricingTable, describePricingTier, fallbackCostLabel, parseCostKey } from "./deploymentPricing";

describe("parseCostKey", () => {
  it.each([
    ["input_cost_per_token", { charge: "input", modality: "", unit: "token", threshold: null, tier: "standard" }],
    ["output_cost_per_token_flex", { charge: "output", unit: "token", threshold: null, tier: "flex" }],
    ["input_cost_per_token_batches", { charge: "input", tier: "batches" }],
    ["cache_read_input_token_cost", { charge: "cache_read", unit: "token", tier: "standard" }],
    ["explicit_cache_read_input_token_cost", { charge: "explicit_cache_read", unit: "token", tier: "standard" }],
    [
      "cache_read_input_token_cost_above_272k_tokens_priority",
      { charge: "cache_read", threshold: 272_000, tier: "priority" },
    ],
    [
      "cache_creation_input_token_cost_above_12hr_above_272k_tokens_flex",
      { charge: "cache_write", cacheDuration: "12hr", threshold: 272_000, tier: "flex" },
    ],
    ["cache_creation_input_token_cost_above_1hr", { charge: "cache_write", cacheDuration: "1hr", threshold: null }],
    ["cache_read_input_audio_token_cost_priority", { charge: "cache_read", modality: "audio", tier: "priority" }],
    ["input_cost_per_audio_token", { charge: "input", modality: "audio", unit: "token" }],
    ["output_cost_per_reasoning_token", { charge: "output", modality: "reasoning", unit: "token" }],
    [
      "input_cost_per_audio_per_second_above_128k_tokens",
      { charge: "input", modality: "audio", unit: "second", threshold: 128_000 },
    ],
    ["output_cost_per_second", { charge: "output", modality: "", unit: "second" }],
    ["input_cost_per_image", { charge: "input", modality: "", unit: "image" }],
    ["input_cost_per_image_token", { charge: "input", modality: "image", unit: "token" }],
    ["output_cost_per_pixel", { unit: "pixel" }],
    ["input_cost_per_character", { unit: "character" }],
    ["input_cost_per_query", { unit: "query" }],
    ["input_cost_per_token_above_200k_tokens_ultrafast", { threshold: 200_000, tier: "ultrafast" }],
    ["output_cost_per_token_balanced", { tier: "balanced" }],
  ])("reads %s", (key, expected) => {
    expect(parseCostKey(key)).toMatchObject({ kind: "price", ...expected });
  });

  it.each([
    "output_cost_per_second_720p",
    "output_cost_per_image_1024",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_above_272k_tokens_decisions",
    "input_cost_per_video_per_second_above_8s_interval",
    "ocr_cost_per_page",
    "input_dbu_cost_per_token",
    "input_cost_per_text_message",
  ])("keeps %s as an unknown key rather than guessing", (key) => {
    expect(parseCostKey(key)).toEqual({ kind: "unknown", key });
  });
});

const CONTRACT_COSTS = {
  input_cost_per_token: 2.5e-6,
  output_cost_per_token: 1e-5,
  cache_read_input_token_cost: 1.25e-6,
  cache_creation_input_token_cost: 3e-6,
  input_cost_per_token_priority: 5e-6,
  input_cost_per_token_flex: 1.25e-6,
  input_cost_per_token_above_200k_tokens: 5e-6,
  cache_read_input_token_cost_above_272k_tokens_priority: 1e-6,
  input_cost_per_token_batches: 1.25e-6,
  output_cost_per_second: 0.001,
  input_cost_per_image: 0.002,
};

describe("buildPricingTable", () => {
  const table = buildPricingTable(CONTRACT_COSTS);

  it("offers a column per service tier the deployment prices, in a fixed order", () => {
    expect(table.tiers).toEqual(["standard", "flex", "priority", "batches"]);
  });

  it("groups prices by context threshold, base prices first", () => {
    expect(table.groups.map((group) => group.threshold)).toEqual([null, 200_000, 272_000]);
  });

  it("puts every tier of one charge on one row, token prices scaled to 1M tokens", () => {
    expect(table.groups[0].rows).toEqual([
      { label: "Input / 1M tokens", prices: { standard: "$2.50", flex: "$1.25", priority: "$5.00", batches: "$1.25" } },
      { label: "Input / image", prices: { standard: "$0.002" } },
      { label: "Cache read / 1M tokens", prices: { standard: "$1.25" } },
      { label: "Cache write / 1M tokens", prices: { standard: "$3.00" } },
      { label: "Output / 1M tokens", prices: { standard: "$10.00" } },
      { label: "Output / second", prices: { standard: "$0.001" } },
    ]);
  });

  it("shows an explicit cached read per 1M tokens on its own row", () => {
    const { groups, unknown } = buildPricingTable({
      cache_read_input_token_cost: 1e-7,
      explicit_cache_read_input_token_cost: 5e-8,
    });
    expect(unknown).toEqual([]);
    expect(groups[0].rows).toEqual([
      { label: "Cache read / 1M tokens", prices: { standard: "$0.10" } },
      { label: "Explicit cache read / 1M tokens", prices: { standard: "$0.05" } },
    ]);
  });

  it("answers a cached read on priority above a threshold from its own group", () => {
    expect(table.groups[2].rows).toEqual([{ label: "Cache read / 1M tokens", prices: { priority: "$1.00" } }]);
  });

  it("lists keys it cannot read under their raw name with the unscaled price", () => {
    const { groups, unknown } = buildPricingTable({ input_cost_per_token: 1e-6, output_cost_per_second_720p: 0.05 });
    expect(unknown).toEqual([{ key: "output_cost_per_second_720p", label: "Output / second 720p", price: "$0.05" }]);
    expect(groups.flatMap((group) => group.rows.map((row) => row.label))).toEqual(["Input / 1M tokens"]);
  });

  it("shows a zero price as free and keeps precision for tiny per-unit prices", () => {
    const { groups } = buildPricingTable({ input_cost_per_token: 0, output_cost_per_pixel: 1e-8 });
    expect(groups[0].rows).toEqual([
      { label: "Input / 1M tokens", prices: { standard: "Free" } },
      { label: "Output / pixel", prices: { standard: "$0.00000001" } },
    ]);
  });

  it("labels modality and cache duration in the row", () => {
    const { groups } = buildPricingTable({
      cache_creation_input_token_cost_above_1hr: 6e-6,
      input_cost_per_audio_token: 4e-5,
      output_cost_per_reasoning_token: 2e-5,
    });
    expect(groups[0].rows.map((row) => row.label)).toEqual([
      "Audio input / 1M tokens",
      "Cache write (1hr) / 1M tokens",
      "Reasoning output / 1M tokens",
    ]);
  });

  it("returns an empty table for a deployment with no prices", () => {
    const empty = { tiers: [], groups: [], hasBlankCells: false, unknown: [] };
    expect(buildPricingTable({})).toEqual(empty);
  });

  it("flags blank cells only when some charge lacks a price in a tier the deployment offers", () => {
    const everyTierPriced = {
      input_cost_per_token: 1e-6,
      input_cost_per_token_priority: 2e-6,
      output_cost_per_token: 3e-6,
      output_cost_per_token_priority: 6e-6,
    };
    expect(buildPricingTable(CONTRACT_COSTS).hasBlankCells).toBe(true);
    expect(buildPricingTable(everyTierPriced).hasBlankCells).toBe(false);
  });
});

describe("fallbackCostLabel", () => {
  it.each([
    ["google_maps_grounding_cost_per_query", "Google maps grounding / query"],
    ["file_search_cost_per_gb_per_day", "File search / gb per day"],
    ["cost_per_second", "Cost / second"],
    ["output_cost_per_second_720p", "Output / second 720p"],
    ["explicit_cache_read_input_token_cost", "Explicit cache read input token"],
    ["some_new_price", "Some new price"],
  ])("names %s as %s", (key, label) => {
    expect(fallbackCostLabel(key)).toBe(label);
  });
});

describe("describePricingTier", () => {
  it("states the token range and each price per 1M tokens, input before output", () => {
    expect(
      describePricingTier({
        range: [0, 256_000],
        costs: { output_cost_per_token: 4e-7, input_cost_per_token: 5e-8 },
      }),
    ).toBe("0–256K tokens: $0.05 input / $0.40 output");
  });

  it("names extra charges and non-token units", () => {
    expect(
      describePricingTier({
        range: [256_000, 1_000_000],
        costs: {
          input_cost_per_token: 2.5e-7,
          output_cost_per_reasoning_token: 1.2e-5,
          input_cost_per_query: 0.01,
          web_search_cost_per_call: 0.02,
        },
      }),
    ).toBe("256K–1M tokens: $0.25 input / $0.01 input per query / $12.00 reasoning output / $0.02 web search / call");
  });
});
