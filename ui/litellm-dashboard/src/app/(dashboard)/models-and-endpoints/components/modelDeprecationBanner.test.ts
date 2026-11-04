import { describe, expect, it } from "vitest";
import type { ModelDeprecationInfo } from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import { deprecationBannerTitle, deprecationEntryText } from "./modelDeprecationBanner";

const makeInfo = (overrides: Partial<ModelDeprecationInfo> = {}): ModelDeprecationInfo => ({
  model_name: "gpt-4o",
  deprecation_date: "2026-11-30",
  days_until_deprecation: 26,
  status: "imminent",
  ...overrides,
});

describe("deprecationBannerTitle", () => {
  it("formats both counts with singular and plural wording", () => {
    expect(deprecationBannerTitle(2, 1, 30)).toBe("2 models are deprecated, 1 deprecates within 30 days");
    expect(deprecationBannerTitle(1, 2, 30)).toBe("1 model is deprecated, 2 deprecate within 30 days");
  });

  it("omits empty counts and uses singular model wording", () => {
    expect(deprecationBannerTitle(1, 0, 30)).toBe("1 model is deprecated");
    expect(deprecationBannerTitle(0, 1, 30)).toBe("1 model deprecates within 30 days");
    expect(deprecationBannerTitle(0, 3, 30)).toBe("3 models deprecate within 30 days");
    expect(deprecationBannerTitle(0, 0, 30)).toBe("");
  });
});

describe("deprecationEntryText", () => {
  it("formats deprecated entries with provider and successor", () => {
    const deprecatedInfo: ModelDeprecationInfo = {
      model_name: "gpt-4",
      litellm_provider: "openai",
      deprecation_date: "2026-06-01",
      days_until_deprecation: -152,
      status: "deprecated",
      successor_model: "gpt-5.6-sol",
    };

    expect(deprecationEntryText(deprecatedInfo)).toBe(
      "gpt-4 (openai) was deprecated on 2026-06-01, migrate to gpt-5.6-sol",
    );
  });

  it("omits a null provider and successor", () => {
    expect(
      deprecationEntryText(
        makeInfo({
          litellm_provider: null,
          successor_model: null,
        }),
      ),
    ).toBe("gpt-4o deprecates on 2026-11-30, in 26 days");
  });

  it("uses singular wording for one day", () => {
    expect(
      deprecationEntryText(
        makeInfo({
          days_until_deprecation: 1,
        }),
      ),
    ).toBe("gpt-4o deprecates on 2026-11-30, in 1 day");
  });
});
