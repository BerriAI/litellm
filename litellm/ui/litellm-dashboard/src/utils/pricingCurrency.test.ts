import { describe, expect, it } from "vitest";

import {
  configuredPrices,
  currencyPatch,
  pricePreview,
  priceUnit,
  rateProblem,
  reinterpretedFields,
  storedCurrencySetting,
  usdRate,
  withoutDerivedPricing,
  type CurrencySetting,
} from "./pricingCurrency";

const usd: CurrencySetting = { currency: "USD", rate: "" };
const huf: CurrencySetting = { currency: "HUF", rate: "0.0025" };

describe("priceUnit", () => {
  it.each([
    ["input_cost_per_token", "million_tokens"],
    ["cache_read_input_token_cost", "million_tokens"],
    ["cache_creation_input_token_cost_above_1hr", "million_tokens"],
    ["output_cost_per_image_token", "million_tokens"],
    ["input_cost_per_token_above_200k_tokens", "million_tokens"],
    ["output_cost_per_second", "second"],
    ["input_cost_per_audio_per_second_above_15s_interval", "second"],
    ["output_cost_per_second_720p", "second"],
    ["input_cost_per_image", "image"],
    ["input_cost_per_image_above_128k_tokens", "image"],
    ["search_context_cost_per_query", "query"],
    ["ocr_cost_per_page", "page"],
    ["input_cost_per_character", "million_characters"],
    ["cost_per_ptu_per_hour", "ptu_hour"],
    ["output_cost_per_billable_unit", "unit"],
    ["tiered_pricing[1].input_cost_per_token", "million_tokens"],
    ["off_peak_pricing.output_cost_per_token", "million_tokens"],
    ["output_cost_per_video_clip.720p.5s", "clip"],
    ["search_context_cost_per_query.search_context_size_low", "query"],
  ])("reads %s as priced %s", (field, unit) => {
    expect(priceUnit(field)).toBe(unit);
  });
});

describe("storedCurrencySetting", () => {
  it("reads a currency configured in model_info when litellm_params has none", () => {
    expect(storedCurrencySetting({ model: "m" }, { pricing_currency: "HUF", usd_per_currency_unit: "0.003" })).toEqual({
      currency: "HUF",
      rate: "0.003",
    });
    expect(
      storedCurrencySetting({ pricing_currency: "EUR", usd_per_currency_unit: "1.2" }, { pricing_currency: "HUF" }),
    ).toEqual({ currency: "EUR", rate: "1.2" });
  });
});

describe("rateProblem", () => {
  it.each([
    [usd, null],
    [{ currency: "EUR", rate: "1.08" }, null],
    [{ currency: "EUR", rate: "2e-3" }, null],
    [{ currency: "EUR", rate: "" }, "required"],
    [{ currency: "EUR", rate: "   " }, "required"],
    [{ currency: "EUR", rate: "0" }, "positive"],
    [{ currency: "EUR", rate: "-1" }, "positive"],
    [{ currency: "EUR", rate: "abc" }, "positive"],
    [{ currency: "EUR", rate: "Infinity" }, "positive"],
  ] as const)("judges %j as %s", (setting, problem) => {
    expect(rateProblem(setting)).toBe(problem);
  });
});

describe("currencyPatch", () => {
  it.each([
    ["an untouched USD model", usd, usd, {}],
    ["an untouched HUF model", huf, { ...huf }, {}],
    [
      "a USD model moved to EUR",
      usd,
      { currency: "EUR", rate: "1.08" },
      { pricing_currency: "EUR", usd_per_currency_unit: "1.08" },
    ],
    ["a new HUF rate", huf, { currency: "HUF", rate: "0.003" }, { usd_per_currency_unit: "0.003" }],
    [
      "a HUF model moved to EUR",
      huf,
      { currency: "EUR", rate: "0.0025" },
      { pricing_currency: "EUR", usd_per_currency_unit: "0.0025" },
    ],
    ["a HUF model moved back to USD", huf, usd, { pricing_currency: "USD", usd_per_currency_unit: null }],
  ] as const)("sends only what changed for %s", (_, stored, edited, patch) => {
    expect(currencyPatch(stored, edited)).toEqual(patch);
  });
});

describe("configuredPrices", () => {
  it("takes the configured fields from usd_pricing and pricing_overrides, litellm_params first", () => {
    expect(
      configuredPrices({
        litellm_params: { input_cost_per_token: 0.0008, output_cost_per_second: "x" },
        model_info: {
          input_cost_per_token: 0.0004,
          cache_read_input_token_cost: 0.00008,
          max_tokens: 10,
          usd_pricing: { input_cost_per_token: 2e-6, output_cost_per_second: 1 },
          pricing_overrides: ["cache_read_input_token_cost"],
        },
      }),
    ).toEqual({ input_cost_per_token: 0.0008, cache_read_input_token_cost: 0.00008 });
  });
});

describe("configuredPrices for price tables", () => {
  it("lists each rate inside tier, off-peak, clip and search tables and skips their settings", () => {
    expect(
      configuredPrices({
        litellm_params: {
          tiered_pricing: [{ range: [0, 32000], input_cost_per_token: 0.000001, output_cost_per_token: null }],
        },
        model_info: {
          off_peak_pricing: { hours_utc: "00:00-06:00", windows: [], input_cost_per_token: 0.0000005 },
          output_cost_per_video_clip: { "720p": { "5s": 0.4 } },
          usd_pricing: { tiered_pricing: [], off_peak_pricing: {}, output_cost_per_video_clip: {} },
        },
      }),
    ).toEqual({
      "tiered_pricing[0].input_cost_per_token": 0.000001,
      "off_peak_pricing.input_cost_per_token": 0.0000005,
      "output_cost_per_video_clip.720p.5s": 0.4,
    });
  });
});

describe("pricePreview", () => {
  it("shows token prices per million and time prices per second, with edits winning", () => {
    const rows = pricePreview(
      { input_cost_per_token: 0.0008, output_cost_per_second: 400, output_cost_per_token: 0.0016 },
      { input_cost_per_token: 0.0012, output_cost_per_token: null },
      huf,
    );
    expect(rows).toEqual([
      { field: "input_cost_per_token", unit: "million_tokens", source: 1200, usd: expect.closeTo(3, 12), edited: true },
      { field: "output_cost_per_second", unit: "second", source: 400, usd: expect.closeTo(1, 12), edited: false },
    ]);
  });

  it("has no USD value until the rate is valid", () => {
    expect(pricePreview({ input_cost_per_token: 1e-6 }, {}, { currency: "EUR", rate: "" })[0].usd).toBeNull();
  });
});

describe("reinterpretedFields", () => {
  const rows = pricePreview(
    { input_cost_per_token: 3e-6, output_cost_per_token: 6e-6 },
    { input_cost_per_token: 1e-6 },
    huf,
  );

  it("names the untouched prices a currency or rate change re-reads", () => {
    expect(reinterpretedFields(rows, usd, huf)).toEqual(["output_cost_per_token"]);
    expect(reinterpretedFields(rows, huf, { currency: "HUF", rate: "0.003" })).toEqual(["output_cost_per_token"]);
  });

  it("names nothing while the currency and rate stay as saved", () => {
    expect(reinterpretedFields(rows, huf, { currency: "HUF", rate: " 0.0025 " })).toEqual([]);
  });
});

describe("derived pricing", () => {
  it("prefers the server's USD rate and never echoes the derived blocks", () => {
    const modelInfo = { usd_pricing: { input_cost_per_token: 2e-6 }, pricing_source: {}, team_id: "t" };
    expect(usdRate(modelInfo, "input_cost_per_token", 0.0008)).toBe(2e-6);
    expect(usdRate(modelInfo, "output_cost_per_token", 4e-6)).toBe(4e-6);
    expect(withoutDerivedPricing(modelInfo)).toEqual({ team_id: "t" });
  });
});
