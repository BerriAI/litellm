import { describe, expect, it } from "vitest";
import { lensEntityHref } from "./entityLinks";

describe("Lens entity links", () => {
  it("opens traces for the selected team without interpreting its identifier as URL syntax", () => {
    const href = lensEntityHref({ kind: "team", teamId: "team/with?query&characters" });
    const url = new URL(href!, "http://localhost");

    expect(url.pathname).toBe("/ui/lens");
    expect(url.searchParams.get("tab")).toBe("traces");
    expect(url.searchParams.get("team_id")).toBe("team/with?query&characters");
    expect(url.searchParams.has("api_key_hash")).toBe(false);
  });

  it("keeps both the team and the original key hash selected", () => {
    const hash = "AB".repeat(32);
    const href = lensEntityHref({ kind: "key", teamId: "team-one", keyHash: hash });
    const url = new URL(href!, "http://localhost");

    expect(url.searchParams.get("team_id")).toBe("team-one");
    expect(url.searchParams.get("api_key_hash")).toBe(hash.toLowerCase());
    expect(url.searchParams.get("tab")).toBe("traces");
  });

  it("can open a personal key without adding a team filter", () => {
    const hash = "cd".repeat(32);
    const href = lensEntityHref({ kind: "key", teamId: null, keyHash: hash });
    const url = new URL(href!, "http://localhost");

    expect(url.searchParams.has("team_id")).toBe(false);
    expect(url.searchParams.get("api_key_hash")).toBe(hash);
  });

  it.each([undefined, null, "", "sk-not-a-hash", "key-identifier", "a".repeat(63), "g".repeat(64)])(
    "does not create a misleading link or put a credential in the URL for invalid key reference %s",
    (keyHash) => {
      expect(lensEntityHref({ kind: "key", teamId: "team-one", keyHash })).toBeNull();
    },
  );
});
