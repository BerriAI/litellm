import { describe, expect, it } from "vitest";
import { searchProviderFilterOption } from "./searchProviderFilterOption";

const searxng = { value: "searxng", title: "SearXNG" };
const googlePse = { value: "google_pse", title: "Google PSE" };

describe("searchProviderFilterOption", () => {
  it("matches a provider by its slug", () => {
    expect(searchProviderFilterOption("searxng", searxng)).toBe(true);
    expect(searchProviderFilterOption("searxng", googlePse)).toBe(false);
  });

  it("matches a provider by its display name, case-insensitively", () => {
    expect(searchProviderFilterOption("google pse", googlePse)).toBe(true);
    expect(searchProviderFilterOption("SEARXNG", searxng)).toBe(true);
  });

  it("matches on a partial slug so results narrow as the user types", () => {
    expect(searchProviderFilterOption("sear", searxng)).toBe(true);
    expect(searchProviderFilterOption("sear", googlePse)).toBe(false);
  });

  it("keeps every option when the query is empty or whitespace", () => {
    expect(searchProviderFilterOption("", googlePse)).toBe(true);
    expect(searchProviderFilterOption("   ", googlePse)).toBe(true);
  });

  it("does not match an option that has no slug or title", () => {
    expect(searchProviderFilterOption("searxng", {})).toBe(false);
    expect(searchProviderFilterOption("searxng", undefined)).toBe(false);
  });
});
