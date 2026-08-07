import { describe, expect, it } from "vitest";

import { normalizeTierModels } from "./complexity_router_tiers";

// The backend types a tier as `str | list[str]` and widens with
// `models if isinstance(models, list) else [models]`
// (litellm/router_strategy/complexity_router/config.py:255, :441). These cases assert the
// expected verdict per input rather than just agreement between call sites, so the test still
// has teeth if every reader were changed at once.
describe("normalizeTierModels", () => {
  it("widens a pinned single model to a one-element pool", () => {
    expect(normalizeTierModels("gpt-4o-mini")).toEqual(["gpt-4o-mini"]);
  });

  it("passes a pool through in order", () => {
    expect(normalizeTierModels(["a", "b"])).toEqual(["a", "b"]);
  });

  it("treats an empty string as no models, not a pool containing an empty name", () => {
    expect(normalizeTierModels("")).toEqual([]);
  });

  it("drops non-string entries rather than typing them as models", () => {
    expect(normalizeTierModels(["a", 3, null, "b"])).toEqual(["a", "b"]);
  });

  it.each([[undefined], [null], [{}], [42]])("returns no models for %s", (value) => {
    expect(normalizeTierModels(value)).toEqual([]);
  });
});
