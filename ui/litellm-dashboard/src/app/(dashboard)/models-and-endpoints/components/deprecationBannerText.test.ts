import { describe, expect, it } from "vitest";
import type { ModelDeprecationInfo } from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import {
  bucketByLocalDate,
  countDistinctModels,
  deprecationBannerTitle,
  deprecationEntryText,
  localDaysUntil,
  parseLocalDate,
} from "./deprecationBannerText";

const makeInfo = (overrides: Partial<ModelDeprecationInfo> = {}): ModelDeprecationInfo => ({
  model_name: "gpt-4o",
  deprecation_date: "2026-11-30",
  days_until_deprecation: 26,
  status: "imminent",
  ...overrides,
});

describe("local deprecation dates", () => {
  it("parses date parts as a local calendar date and counts from local midnight", () => {
    const parsed = parseLocalDate("2026-11-06");

    expect([parsed.getFullYear(), parsed.getMonth(), parsed.getDate(), parsed.getHours()]).toEqual([2026, 10, 6, 0]);
    expect(localDaysUntil("2026-11-06", new Date(2026, 10, 4, 18))).toBe(2);
  });

  it("buckets entries by their local date and warning window", () => {
    const buckets = bucketByLocalDate(
      [
        makeInfo({ model_name: "yesterday", deprecation_date: "2026-11-03", status: "imminent" }),
        makeInfo({ model_name: "today", deprecation_date: "2026-11-04", status: "deprecated" }),
        makeInfo({ model_name: "outside-window", deprecation_date: "2026-12-05", status: "imminent" }),
      ],
      new Date(2026, 10, 4),
      30,
    );

    expect(
      buckets.deprecated.map(({ model_name, days_until_deprecation, status }) => [
        model_name,
        days_until_deprecation,
        status,
      ]),
    ).toEqual([["yesterday", -1, "deprecated"]]);
    expect(
      buckets.imminent.map(({ model_name, days_until_deprecation, status }) => [
        model_name,
        days_until_deprecation,
        status,
      ]),
    ).toEqual([["today", 0, "imminent"]]);
  });
});

describe("countDistinctModels", () => {
  it("counts visible names and providers, not deployment aliases", () => {
    expect(
      countDistinctModels([
        makeInfo({ model_name: "team-a", team_public_model_name: "shared", litellm_provider: "openai" }),
        makeInfo({ model_name: "team-b", team_public_model_name: "shared", litellm_provider: "openai" }),
        makeInfo({ model_name: "team-c", team_public_model_name: "shared", litellm_provider: "anthropic" }),
      ]),
    ).toBe(2);
  });
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

  it("says an imminent model deprecates today", () => {
    expect(
      deprecationEntryText(
        makeInfo({
          days_until_deprecation: 0,
          deprecation_date: "2026-11-04",
        }),
      ),
    ).toBe("gpt-4o deprecates today (2026-11-04)");
  });

  it("uses the team-visible model name when present", () => {
    expect(
      deprecationEntryText(
        makeInfo({
          model_name: "internal-name",
          team_public_model_name: "team-name",
        }),
      ),
    ).toBe("team-name deprecates on 2026-11-30, in 26 days");
  });
});
