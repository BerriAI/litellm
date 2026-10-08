import { describe, expect, it } from "vitest";

import slackLogo from "../../../../../public/assets/logos/slack.svg";
import { sourceApp } from "./RunSource";

describe("sourceApp", () => {
  it.each([
    ["slack", "https://acme.slack.com/archives/C1/p1", "Slack"],
    ["teams", "https://teams.microsoft.com/l/message/19:abc/1", "Teams"],
    ["discord", "https://discord.com/channels/1/2/3", "Discord"],
    ["linear", "https://linear.app/acme/issue/LIT-1", "Linear"],
    ["github", "https://github.com/BerriAI/litellm/issues/1", "GitHub"],
    ["jira", "https://acme.atlassian.net/browse/LIT-1", "Jira"],
  ] as const)("brands a %s url on its own domain", (type, url, label) => {
    expect(sourceApp({ type, url })?.label).toBe(label);
  });

  it("shows the Slack logo for a slack.com thread", () => {
    expect(sourceApp({ type: "slack", url: "https://acme.slack.com/archives/C1/p1" })?.logo).toBe(slackLogo.src);
  });

  it.each([
    ["off-domain url", "https://attacker.example/login", "attacker.example"],
    ["lookalike suffix", "https://slack.com.attacker.example/x", "slack.com.attacker.example"],
    ["lookalike prefix", "https://evilslack.com/x", "evilslack.com"],
  ])("does not brand a slack source with an %s", (_, url, hostname) => {
    expect(sourceApp({ type: "slack", url })).toEqual({ label: hostname, logo: null });
  });

  it("shows a custom source's hostname", () => {
    expect(sourceApp({ type: "custom", url: "https://bot.acme.dev/c/42" })).toEqual({
      label: "bot.acme.dev",
      logo: null,
    });
  });

  it.each(["javascript:alert(1)", "http://acme.slack.com/archives/C1/p1", "not a url", ""])("rejects %s", (bad) => {
    expect(sourceApp({ type: "slack", url: bad })).toBeNull();
  });
});
