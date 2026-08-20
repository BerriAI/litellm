#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

const NEUTRALS = ["slate", "gray", "zinc", "neutral", "stone"];

const SEMANTICS = {
  destructive: ["red", "rose"],
  success: ["green", "emerald"],
  warning: ["amber", "yellow", "orange"],
  info: ["blue", "sky", "cyan"],
};

const NEUTRAL_RULES = {
  text: {
    950: "text-foreground",
    900: "text-foreground",
    800: "text-foreground",
    700: "text-foreground",
    600: "text-muted-foreground",
    500: "text-muted-foreground",
    400: "text-muted-foreground/70",
    300: "text-muted-foreground/70",
  },
  bg: {
    50: "bg-muted",
    100: "bg-muted",
    200: "bg-border",
    300: "bg-border",
    400: "bg-border",
  },
  border: {
    100: "border-border",
    200: "border-border",
    300: "border-border",
    400: "border-border",
  },
  divide: { 100: "divide-border", 200: "divide-border", 300: "divide-border" },
  ring: { 100: "ring-border", 200: "ring-border", 300: "ring-border" },
  placeholder: {
    300: "placeholder-muted-foreground/70",
    400: "placeholder-muted-foreground/70",
    500: "placeholder-muted-foreground",
  },
};

const semanticRules = (token) => ({
  text: Object.fromEntries([300, 400, 500, 600, 700, 800, 900, 950].map((shade) => [shade, `text-${token}`])),
  bg: {
    50: `bg-${token}/10`,
    100: `bg-${token}/15`,
    200: `bg-${token}/20`,
    300: `bg-${token}/30`,
    400: `bg-${token}`,
    500: `bg-${token}`,
    600: `bg-${token}`,
    700: `bg-${token}`,
    900: `bg-${token}/15`,
    950: `bg-${token}/10`,
  },
  border: {
    100: `border-${token}/15`,
    200: `border-${token}/20`,
    300: `border-${token}/30`,
    400: `border-${token}`,
    500: `border-${token}`,
    600: `border-${token}`,
    700: `border-${token}`,
    800: `border-${token}/30`,
  },
  divide: { 100: `divide-${token}/15`, 200: `divide-${token}/20`, 300: `divide-${token}/30` },
  fill: Object.fromEntries([400, 500, 600, 700].map((shade) => [shade, `fill-${token}`])),
  stroke: Object.fromEntries([400, 500, 600, 700].map((shade) => [shade, `stroke-${token}`])),
});

const LITERALS = {
  "bg-white": "bg-card",
  "text-black": "text-foreground",
  "ring-blue-400": "ring-ring",
  "ring-blue-500": "ring-ring",
  "ring-blue-600": "ring-ring",
  "ring-offset-white": "ring-offset-background",
};

const buildMapping = () => {
  const entries = Object.entries(LITERALS);
  const expand = (families, rules) =>
    families.flatMap((family) =>
      Object.entries(rules).flatMap(([prefix, shades]) =>
        Object.entries(shades).map(([shade, replacement]) => [`${prefix}-${family}-${shade}`, replacement]),
      ),
    );
  return new Map([
    ...entries,
    ...expand(NEUTRALS, NEUTRAL_RULES),
    ...Object.entries(SEMANTICS).flatMap(([token, families]) => expand(families, semanticRules(token))),
  ]);
};

const MAPPING = buildMapping();

const SENTINEL = "\u0000";
const VARIANT = String.raw`(?:[a-z][a-z0-9-]*(?:\[[^\]\s]*\])?:)`;
const CORES = [...MAPPING.keys()].sort((a, b) => b.length - a.length).join("|");
const UTILITY = new RegExp(String.raw`(?<![\w:/[-])(${VARIANT}*)(!?)(${CORES})(?![\w/[-])`, "g");

const ALL_FAMILIES = [
  ...NEUTRALS,
  ...Object.values(SEMANTICS).flat(),
  "lime",
  "teal",
  "indigo",
  "violet",
  "purple",
  "fuchsia",
  "pink",
  "white",
  "black",
];
const SWATCH = String.raw`(?:bg|text|border|ring|divide|from|to|via|placeholder|decoration|outline|fill|stroke|accent|caret|shadow)-(?:${ALL_FAMILIES.join("|")})(?:-(?:50|100|200|300|400|500|600|700|800|900|950))?`;
const PALETTE = new RegExp(String.raw`(?<![\w:/[-])${VARIANT}*!?${SWATCH}(?![\w/[-])`, "g");
const DARK_PALETTE = new RegExp(
  String.raw`(?<![\w:/[-])${VARIANT}*dark:${VARIANT}*!?${SWATCH}(?:/\d+)?(?![\w[-])`,
  "g",
);

const INTERACTION = /\b(?:hover|focus|focus-visible|focus-within|active|selected)\b/;
const LIFT_ON_INTERACTION = new Set(["bg-muted", "bg-border"]);

const stripDark = !process.argv.includes("--keep-dark");
const dryRun = process.argv.includes("--dry");

const files = execFileSync("git", ["ls-files", "src"], { encoding: "utf8" })
  .split("\n")
  .filter((file) => file.endsWith(".ts") || file.endsWith(".tsx"));

const tally = (counts, key) => new Map(counts).set(key, (counts.get(key) ?? 0) + 1);

const rewrite = (source, seed) => {
  const stats = { applied: seed.applied, dropped: seed.dropped };
  const darkStripped = stripDark
    ? source.replace(DARK_PALETTE, (match) => {
        stats.dropped = tally(stats.dropped, match);
        return SENTINEL;
      })
    : source;
  const replaced = darkStripped.replace(UTILITY, (match, variants, important, core) => {
    const mapped = MAPPING.get(core);
    const token = INTERACTION.test(variants) && LIFT_ON_INTERACTION.has(mapped) ? "bg-accent" : mapped;
    stats.applied = tally(stats.applied, `${variants}${core} -> ${variants}${token}`);
    return `${variants}${important}${token}`;
  });
  const cleaned = replaced.replaceAll(`${SENTINEL} `, "").replace(new RegExp(` ?${SENTINEL}`, "g"), "");
  return { text: cleaned, ...stats };
};

const result = files.reduce(
  (acc, file) => {
    const source = readFileSync(file, "utf8");
    const { text, applied, dropped } = rewrite(source, acc);
    const leftovers = [...text.matchAll(PALETTE)].reduce((counts, hit) => tally(counts, hit[0]), acc.leftovers);
    if (text === source) return { ...acc, applied, dropped, leftovers };
    if (!dryRun) writeFileSync(file, text);
    return { applied, dropped, leftovers, changed: [...acc.changed, file] };
  },
  { applied: new Map(), dropped: new Map(), leftovers: new Map(), changed: [] },
);

const { leftovers } = result;

const sum = (counts) => [...counts.values()].reduce((total, value) => total + value, 0);
const byCount = (a, b) => b[1] - a[1];

console.log(`${dryRun ? "[dry run] " : ""}files changed: ${result.changed.length}/${files.length}`);
console.log(`utilities rewritten: ${sum(result.applied)}`);
console.log(`redundant dark: variants dropped: ${sum(result.dropped)}`);
console.log("\nrules applied:");
for (const [rule, count] of [...result.applied].sort(byCount)) console.log(`  ${String(count).padStart(4)}  ${rule}`);
console.log(`\nunmapped palette utilities remaining (${sum(leftovers)} occurrences):`);
for (const [util, count] of [...leftovers].sort(byCount)) console.log(`  ${String(count).padStart(4)}  ${util}`);
