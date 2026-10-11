import { Box, Sparkles } from "lucide-react";
import { describe, expect, it } from "vitest";
import { latestLaunches, launchIcon, type WhatsNewLaunch } from "./homeContent";

const launch = (title: string, published_on: string): WhatsNewLaunch => ({
  icon: "box",
  title,
  description: "d",
  href: `https://docs.litellm.ai/${title}`,
  published_on,
});

describe("latestLaunches", () => {
  it("orders launches by published_on, newest first, without mutating the input", () => {
    const launches = [launch("old", "2026-09-01"), launch("new", "2026-10-11"), launch("mid", "2026-10-01")];
    expect(latestLaunches(launches).map((l) => l.title)).toEqual(["new", "mid", "old"]);
    expect(launches.map((l) => l.title)).toEqual(["old", "new", "mid"]);
  });

  it("keeps only the three newest so the cards fill one row", () => {
    const launches = [
      launch("oldest", "2026-08-01"),
      launch("a", "2026-10-03"),
      launch("b", "2026-10-02"),
      launch("c", "2026-10-01"),
    ];
    expect(latestLaunches(launches).map((l) => l.title)).toEqual(["a", "b", "c"]);
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
