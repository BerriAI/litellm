import type { ColumnFiltersState } from "@tanstack/react-table";
import { describe, expect, it } from "vitest";

import {
  FEATURE_FILTER_ID,
  MODE_FILTER_ID,
  PROVIDER_FILTER_ID,
  featureLabel,
  readFilterValues,
  serializePublicModelHubFilters,
  withFilterValue,
} from "./publicModelHubFilters";

describe("serializePublicModelHubFilters", () => {
  it("sends each multi-select as the route's comma separated in filter", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: ["chat", "embedding"] },
      { id: PROVIDER_FILTER_ID, value: ["openai", "anthropic"] },
      { id: FEATURE_FILTER_ID, value: ["vision"] },
    ];

    expect(serializePublicModelHubFilters(filters)).toEqual({
      "filter[mode][in]": "chat,embedding",
      "filter[providers][in]": "openai,anthropic",
      "filter[features][in]": "vision",
    });
  });

  it("omits blank filters rather than sending parameters the route rejects", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: [] },
      { id: PROVIDER_FILTER_ID, value: [] },
    ];

    expect(serializePublicModelHubFilters(filters)).toEqual({});
  });

  it("ignores filter ids the route does not declare", () => {
    expect(serializePublicModelHubFilters([{ id: "health_status", value: ["healthy"] }])).toEqual({});
  });
});

describe("readFilterValues", () => {
  it("reads back the values of the filter it names", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: ["chat"] },
      { id: FEATURE_FILTER_ID, value: ["vision", "reasoning"] },
    ];

    expect(readFilterValues(filters, FEATURE_FILTER_ID)).toEqual(["vision", "reasoning"]);
    expect(readFilterValues(filters, PROVIDER_FILTER_ID)).toEqual([]);
  });
});

describe("withFilterValue", () => {
  it("adds a filter that is not set yet", () => {
    expect(withFilterValue([], PROVIDER_FILTER_ID, ["openai"])).toEqual([
      { id: PROVIDER_FILTER_ID, value: ["openai"] },
    ]);
  });

  it("replaces a filter instead of stacking a second one", () => {
    const filters: ColumnFiltersState = [{ id: PROVIDER_FILTER_ID, value: ["openai"] }];

    expect(withFilterValue(filters, PROVIDER_FILTER_ID, ["anthropic"])).toEqual([
      { id: PROVIDER_FILTER_ID, value: ["anthropic"] },
    ]);
  });

  it("drops a cleared filter and leaves the others alone", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: ["chat"] },
      { id: PROVIDER_FILTER_ID, value: ["openai"] },
    ];

    expect(withFilterValue(filters, PROVIDER_FILTER_ID, [])).toEqual([{ id: MODE_FILTER_ID, value: ["chat"] }]);
  });
});

describe("featureLabel", () => {
  it("renders a route feature the way the hub has always labelled it", () => {
    expect(featureLabel("vision")).toBe("Vision");
    expect(featureLabel("parallel_function_calling")).toBe("Parallel Function Calling");
  });
});
