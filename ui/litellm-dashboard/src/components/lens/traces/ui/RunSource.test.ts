import { describe, expect, it } from "vitest";

import slackLogo from "../../../../../public/assets/logos/slack.svg";
import { sourceApp } from "./RunSource";

const url = "https://acme.slack.com/archives/C1/p1";

describe("sourceApp", () => {
  it.each([
    ["slack", "Slack"],
    ["teams", "Teams"],
    ["discord", "Discord"],
    ["linear", "Linear"],
    ["github", "GitHub"],
    ["jira", "Jira"],
    ["custom", "Conversation"],
  ] as const)("labels type %s as %s", (type, label) => {
    expect(sourceApp({ type, url })?.label).toBe(label);
  });

  it("shows the Slack logo for slack sources", () => {
    expect(sourceApp({ type: "slack", url })?.logo).toBe(slackLogo.src);
  });

  it("uses the declared type, not the url host", () => {
    expect(sourceApp({ type: "custom", url })?.label).toBe("Conversation");
  });

  it.each(["javascript:alert(1)", "http://acme.slack.com/archives/C1/p1", "not a url", ""])("rejects %s", (bad) => {
    expect(sourceApp({ type: "slack", url: bad })).toBeNull();
  });
});
