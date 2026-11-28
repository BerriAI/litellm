import { describe, expect, it } from "vitest";
import { memberImportBudget, parseMemberEmails } from "./bulkMembers";

describe("parseMemberEmails", () => {
  it("accepts pasted delimiters and skips duplicates and existing members without changing their roles", () => {
    expect(
      parseMemberEmails("New@example.com; other@example.com\nNEW@example.com, old@example.com invalid", [
        "OLD@example.com",
      ]),
    ).toEqual([
      { email: "New@example.com", status: "ready" },
      { email: "other@example.com", status: "ready" },
      { email: "NEW@example.com", status: "duplicate" },
      { email: "old@example.com", status: "existing" },
      { email: "invalid", status: "invalid" },
    ]);
  });
  it("reads a quoted CSV column with an optional header and blank lines", () => {
    expect(parseMemberEmails('user_email\r\n"a@example.com"\r\n\r\nb@example.com', [])).toEqual([
      { email: "a@example.com", status: "ready" },
      { email: "b@example.com", status: "ready" },
    ]);
    expect(parseMemberEmails("", [])).toEqual([]);
  });
});

describe("memberImportBudget", () => {
  it("preserves optional and zero budgets and rejects invalid limits", () => {
    expect(memberImportBudget("")).toBeUndefined();
    expect(memberImportBudget("0")).toBe(0);
    expect(memberImportBudget("12.5")).toBe(12.5);
    expect(memberImportBudget("-1")).toBe("invalid");
    expect(memberImportBudget("Infinity")).toBe("invalid");
    expect(memberImportBudget("bad")).toBe("invalid");
  });
});
