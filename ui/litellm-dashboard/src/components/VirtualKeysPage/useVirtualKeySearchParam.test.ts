import { describe, expect, it } from "vitest";
import { VIRTUAL_KEY_PARAM } from "./useVirtualKeySearchParam";

describe("VIRTUAL_KEY_PARAM", () => {
  it("uses virtual_key as the Virtual Keys page query param", () => {
    expect(VIRTUAL_KEY_PARAM).toBe("virtual_key");
  });
});
