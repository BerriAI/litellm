import { describe, expect, it } from "vitest";

import { sourceApp } from "./RunSource";

describe("sourceApp", () => {
  it.each([
    ["https://acme.slack.com/archives/C1/p1", "Slack"],
    ["https://linear.app/acme/issue/LIT-1", "Linear"],
    ["https://github.com/BerriAI/litellm/issues/1", "GitHub"],
    ["https://www.notion.so/page", "Notion"],
    ["https://tickets.example.com/42", "tickets.example.com"],
  ])("labels %s as %s", (url, label) => {
    expect(sourceApp(url)?.label).toBe(label);
  });

  it("does not trust a lookalike host", () => {
    expect(sourceApp("https://slack.com.evil.example/x")?.label).toBe("slack.com.evil.example");
  });

  it.each(["javascript:alert(1)", "http://acme.slack.com/archives/C1/p1", "not a url", ""])(
    "rejects %s",
    (url) => {
      expect(sourceApp(url)).toBeNull();
    },
  );
});
