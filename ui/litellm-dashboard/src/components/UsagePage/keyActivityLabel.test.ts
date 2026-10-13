import { keyActivityLabel } from "./keyActivityLabel";

describe("keyActivityLabel", () => {
  it("prefers key_alias", () => {
    expect(keyActivityLabel({ key_alias: "batch-worker", user_email: "alice@example.com" })).toBe("batch-worker");
  });

  it("falls back to user_email when alias is missing", () => {
    expect(keyActivityLabel({ key_alias: null, user_email: "alice@example.com" })).toBe("alice@example.com");
  });

  it("prefers the user alias over the user email", () => {
    expect(keyActivityLabel({ key_alias: null, user_alias: "Alice A", user_email: "alice@example.com" })).toBe(
      "Alice A",
    );
  });

  it("uses the fallback when both alias and email are missing", () => {
    expect(keyActivityLabel({ key_alias: null, user_email: null }, "key-hash-abc")).toBe("key-hash-abc");
  });
});
