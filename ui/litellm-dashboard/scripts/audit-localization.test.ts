import { describe, expect, it } from "vitest";
import { auditSource } from "./audit-localization.mjs";

describe("localization audit", () => {
  it("reports raw user-facing JSX copy", () => {
    expect(auditSource("return <Button>Save changes</Button>", "fixture.tsx")).toEqual([
      expect.objectContaining({ text: "Save changes" }),
    ]);
  });

  it("allows documented technical literals", () => {
    expect(auditSource("return <code>MASTER_KEY</code>", "fixture.tsx")).toEqual([]);
  });
});
