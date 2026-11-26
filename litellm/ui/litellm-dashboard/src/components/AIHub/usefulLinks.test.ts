import { describe, expect, it } from "vitest";
import { toUsefulLinksRequest } from "./usefulLinksMapper";
import { createUsefulLinksSchemas } from "./usefulLinksSchema";

const legacy = { id: "legacy", displayName: "Legacy", url: "https://example.test/legacy", index: 0 };
const reference = { id: "reference", displayName: "Reference", url: "https://example.test/ref", index: 4 };
const { addSchema, editSchema } = createUsefulLinksSchemas([legacy, reference]);

describe("Useful link drafts", () => {
  it("accepts a URL supported by the URL parser without rewriting the input", () => {
    const draft = { displayName: "Docs", url: "  mailto:docs@example.test  " };
    expect(addSchema.parse(draft)).toEqual(draft);
  });

  it("requires both fields when adding a link", () => {
    expect(addSchema.safeParse({ displayName: "", url: legacy.url }).error?.issues[0]?.message).toBe("missingFields");
    expect(addSchema.safeParse({ displayName: "Docs", url: "" }).error?.issues[0]?.message).toBe("missingFields");
  });

  it("checks the URL before the duplicate name", () => {
    const result = addSchema.safeParse({ displayName: legacy.displayName, url: "/not-absolute" });
    expect(result.error?.issues[0]?.message).toBe("invalidUrl");
  });

  it("rejects exact duplicate names but accepts a case variant", () => {
    expect(addSchema.safeParse(legacy).error?.issues[0]?.message).toBe("duplicateName");
    expect(addSchema.safeParse({ ...legacy, displayName: "legacy" }).success).toBe(true);
  });

  it("accepts the current or empty name on edit but rejects another link's name", () => {
    expect(editSchema.safeParse({ link: legacy }).success).toBe(true);
    expect(editSchema.safeParse({ link: { ...legacy, displayName: "" } }).success).toBe(true);
    const duplicate = editSchema.safeParse({ link: { ...legacy, displayName: reference.displayName } });
    expect(duplicate.error?.issues[0]?.message).toBe("duplicateName");
  });

  it("rejects an invalid edit URL", () => {
    const result = editSchema.safeParse({ link: { ...legacy, url: "not a URL" } });
    expect(result.error?.issues[0]?.message).toBe("invalidUrl");
  });
});

describe("Useful links request", () => {
  it("serializes the full reordered map with fresh indices and untouched URLs", () => {
    const edited = { ...legacy, displayName: "", url: "  https://EXAMPLE.test  " };
    expect(toUsefulLinksRequest([reference, edited])).toEqual({
      useful_links: {
        Reference: { url: reference.url, index: 0 },
        "": { url: edited.url, index: 1 },
      },
    });
  });

  it("sends an empty map when the last link is removed", () => {
    expect(toUsefulLinksRequest([])).toEqual({ useful_links: {} });
  });
});
