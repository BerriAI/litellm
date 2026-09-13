import { describe, expect, it } from "vitest";

import { moveTagsOutOfMetadataJson } from "./keyEditFieldNormalizers";

describe("moveTagsOutOfMetadataJson", () => {
  it("moves a tags array out of the JSON and appends it to the current tags", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": ["pilot-tag"], "env": "non-prod"}', ["ui-tag"])).toEqual({
      metadata: '{\n  "env": "non-prod"\n}',
      tags: ["ui-tag", "pilot-tag"],
      movedTags: ["pilot-tag"],
    });
  });

  it("drops duplicates and non-string entries without reporting them as moved", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": ["a", "a", "ui-tag", 7, null]}', ["ui-tag"])).toEqual({
      metadata: "{}",
      tags: ["ui-tag", "a"],
      movedTags: ["a"],
    });
  });

  it("strips an empty tags array while moving nothing", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": []}', undefined)).toEqual({
      metadata: "{}",
      tags: [],
      movedTags: [],
    });
  });

  it("is a no-op when there is no tags array to move", () => {
    expect(moveTagsOutOfMetadataJson('{"env": "prod"}', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson('{"tags": "not-an-array"}', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson("", ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson(undefined, ["a"])).toBeNull();
  });

  it("leaves invalid or non-object JSON alone so the save path can report it", () => {
    expect(moveTagsOutOfMetadataJson('{"tags": [', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson('["tags"]', ["a"])).toBeNull();
    expect(moveTagsOutOfMetadataJson("null", ["a"])).toBeNull();
  });
});
