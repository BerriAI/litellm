import { describe, expect, it } from "vitest";
import { memberMappings, pageParser } from "./forms";

it.each(["0", "-1", "1.5", "junk", "9999999999999999999999"])("rejects invalid page query %s", (value) => {
  expect(pageParser.parse(value)).toBeNull();
});

describe("member mapping CSV", () => {
  it("normalizes member IDs and LiteLLM user IDs", () => {
    expect(memberMappings(" member-one , user-one \r\nmember-two,user-two\n")).toEqual({
      ok: true,
      mappings: [
        { external_actor_id: "member-one", user_id: "user-one" },
        { external_actor_id: "member-two", user_id: "user-two" },
      ],
    });
  });
  it.each(["", "  \n", "member-only", "member,", ",user", "member,user,extra"])(
    "rejects empty or malformed input %s",
    (csv) => {
      expect(memberMappings(csv)).toEqual({ ok: false, error: expect.any(String) });
    },
  );
  it("rejects batches larger than the API cap", () => {
    expect(memberMappings(Array.from({ length: 1001 }, (_, i) => `member-${i},user`).join("\n"))).toEqual({
      ok: false,
      error: "Paste between 1 and 1,000 member mappings.",
    });
  });
});
