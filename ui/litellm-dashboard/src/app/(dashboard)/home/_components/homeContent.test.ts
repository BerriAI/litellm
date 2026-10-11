import { Box, Sparkles } from "lucide-react";
import { describe, expect, it } from "vitest";
import { launchIcon, newestFirst, type WhatsNewLaunch } from "./homeContent";

const launch = (title: string, published_on: string): WhatsNewLaunch => ({
  icon: "box",
  title,
  description: "d",
  href: `https://docs.litellm.ai/${title}`,
  published_on,
});

describe("newestFirst", () => {
  it("orders launches by published_on, newest first, without mutating the input", () => {
    const launches = [launch("old", "2026-09-01"), launch("new", "2026-10-11"), launch("mid", "2026-10-01")];
    expect(newestFirst(launches).map((l) => l.title)).toEqual(["new", "mid", "old"]);
    expect(launches.map((l) => l.title)).toEqual(["old", "new", "mid"]);
  });
});

describe("launchIcon", () => {
  it("maps a known icon name to its lucide icon", () => {
    expect(launchIcon("box")).toBe(Box);
  });

  it("falls back to Sparkles for an icon name this dashboard version does not know", () => {
    expect(launchIcon("hologram")).toBe(Sparkles);
    expect(launchIcon("constructor")).toBe(Sparkles);
  });
});
