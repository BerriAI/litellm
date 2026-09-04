import type { ColumnFiltersState } from "@tanstack/react-table";
import { describe, expect, it } from "vitest";

import {
  MODE_FILTER_ID,
  PROVIDER_FILTER_ID,
  readModeFilter,
  serializePublicModelHubFilters,
  withFilterValue,
} from "./publicModelHubFilters";

describe("serializePublicModelHubFilters", () => {
  it("sends selected modes as the route's comma separated in filter", () => {
    expect(serializePublicModelHubFilters([{ id: MODE_FILTER_ID, value: ["chat", "embedding"] }])).toEqual({
      "filter[mode][in]": "chat,embedding",
    });
  });

  it("sends a provider as the route's contains filter, trimmed", () => {
    expect(serializePublicModelHubFilters([{ id: PROVIDER_FILTER_ID, value: "  anthropic " }])).toEqual({
      "filter[providers][contains]": "anthropic",
    });
  });

  it("serializes mode and provider together", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: ["chat"] },
      { id: PROVIDER_FILTER_ID, value: "openai" },
    ];

    expect(serializePublicModelHubFilters(filters)).toEqual({
      "filter[mode][in]": "chat",
      "filter[providers][contains]": "openai",
    });
  });

  it("omits blank filters rather than sending parameters the route rejects", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: [] },
      { id: PROVIDER_FILTER_ID, value: "   " },
    ];

    expect(serializePublicModelHubFilters(filters)).toEqual({});
  });

  it("ignores filter ids the route does not declare", () => {
    expect(serializePublicModelHubFilters([{ id: "supports_vision", value: ["true"] }])).toEqual({});
  });
});

describe("readModeFilter", () => {
  it("reads back the selected modes", () => {
    expect(readModeFilter([{ id: MODE_FILTER_ID, value: ["chat", "rerank"] }])).toEqual(["chat", "rerank"]);
  });

  it("is empty when no mode filter is set", () => {
    expect(readModeFilter([{ id: PROVIDER_FILTER_ID, value: "openai" }])).toEqual([]);
  });
});

describe("withFilterValue", () => {
  it("adds a filter that is not set yet", () => {
    expect(withFilterValue([], PROVIDER_FILTER_ID, "openai")).toEqual([{ id: PROVIDER_FILTER_ID, value: "openai" }]);
  });

  it("replaces a filter instead of stacking a second one", () => {
    const filters: ColumnFiltersState = [{ id: PROVIDER_FILTER_ID, value: "openai" }];

    expect(withFilterValue(filters, PROVIDER_FILTER_ID, "anthropic")).toEqual([
      { id: PROVIDER_FILTER_ID, value: "anthropic" },
    ]);
  });

  it("drops a cleared filter and leaves the others alone", () => {
    const filters: ColumnFiltersState = [
      { id: MODE_FILTER_ID, value: ["chat"] },
      { id: PROVIDER_FILTER_ID, value: "openai" },
    ];

    expect(withFilterValue(filters, PROVIDER_FILTER_ID, "  ")).toEqual([{ id: MODE_FILTER_ID, value: ["chat"] }]);
    expect(withFilterValue(filters, MODE_FILTER_ID, [])).toEqual([{ id: PROVIDER_FILTER_ID, value: "openai" }]);
  });
});
