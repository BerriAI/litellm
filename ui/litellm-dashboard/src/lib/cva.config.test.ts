/* eslint-disable local/no-ad-hoc-z-index -- exercises how cn merges the banned numeric classes against the tokens */
import { describe, expect, it } from "vitest";
import { cn } from "./cva.config";

describe("cn z-index token merging", () => {
  it("keeps only the last z token when two tokens conflict", () => {
    expect(cn("z-raised", "z-overlay")).toBe("z-overlay");
  });

  it("lets a z token override a numeric z class", () => {
    expect(cn("z-50", "z-popup")).toBe("z-popup");
  });

  it("lets a numeric z class override a z token", () => {
    expect(cn("z-popup", "z-50")).toBe("z-50");
  });

  it("does not merge z tokens with unrelated classes", () => {
    expect(cn("z-chrome", "sticky", "top-0")).toBe("z-chrome sticky top-0");
  });
});
