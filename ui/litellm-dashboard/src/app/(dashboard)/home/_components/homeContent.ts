import { Bot, Box, Gauge, Plug, Rocket, Scale, Shield, Sparkles, Zap, type LucideIcon } from "lucide-react";
import type { components } from "@/lib/http/schema";

export type WhatsNewLaunch = components["schemas"]["WhatsNewLaunch"];

const LAUNCH_ICONS: ReadonlyMap<string, LucideIcon> = new Map([
  ["bot", Bot],
  ["box", Box],
  ["gauge", Gauge],
  ["plug", Plug],
  ["rocket", Rocket],
  ["scale", Scale],
  ["shield", Shield],
  ["sparkles", Sparkles],
  ["zap", Zap],
]);

export const launchIcon = (name: string): LucideIcon => LAUNCH_ICONS.get(name) ?? Sparkles;

export const newestFirst = (launches: readonly WhatsNewLaunch[]): readonly WhatsNewLaunch[] =>
  [...launches].sort((a, b) => b.published_on.localeCompare(a.published_on));

export const formatPublishedOn = (isoDate: string): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
