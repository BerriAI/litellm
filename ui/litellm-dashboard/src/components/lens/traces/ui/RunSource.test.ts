import { describe, expect, it } from "vitest";

import slackLogo from "../../../../../public/assets/logos/slack.svg";
import { sourceApp } from "./RunSource";

const url = "https://acme.slack.com/archives/C1/p1";

describe("sourceApp", () => {
  it.each([
    ["slack", "Slack thread"],
    ["teams", "Teams thread"],
    ["discord", "Discord thread"],
    ["linear", "Linear issue"],
    ["github", "GitHub thread"],
    ["jira", "Jira issue"],
    ["custom", "Agent conversation"],
  ] as const)("labels type %s as %s", (type, link) => {
    expect(sourceApp({ type, url })?.link).toBe(link);
  });

  it("shows the Slack logo for slack sources", () => {
    expect(sourceApp({ type: "slack", url })?.logo).toBe(slackLogo.src);
  });

  it("uses the declared type, not the url host", () => {
    expect(sourceApp({ type: "custom", url })?.link).toBe("Agent conversation");
  });

  it.each(["javascript:alert(1)", "http://acme.slack.com/archives/C1/p1", "not a url", ""])("rejects %s", (bad) => {
    expect(sourceApp({ type: "slack", url: bad })).toBeNull();
  });
});
