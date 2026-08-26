import { readFileSync } from "fs";
import { resolve } from "path";
import React from "react";
import { describe, it, expect, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import PremiumLoggingSettings from "./PremiumLoggingSettings";

const SOURCE_PATH = resolve(process.cwd(), "src/components/common_components/PremiumLoggingSettings.tsx");

const HARDCODED_PALETTE =
  /\b(?:text|bg|border|hover:bg|hover:text|hover:border|dark:bg|dark:text|dark:border|ring|divide|fill|stroke)-(?:gray|slate|zinc|neutral|stone|red|blue|green|yellow|amber|orange|indigo|purple|pink|rose|teal|cyan|sky|violet|fuchsia|lime|emerald)-\d+(?:\/\d+)?\b/g;

const SEMANTIC_TOKEN =
  /\b(?:text|bg|border|hover:bg|hover:text|ring|divide|fill|stroke)-(?:foreground|muted-foreground|muted|background|card|popover|primary|secondary|destructive|border|input|accent|ring)(?:-foreground)?(?:\/\d+)?\b/g;

describe("PremiumLoggingSettings", () => {
  it("styles itself from semantic tokens instead of hardcoded palette classes", () => {
    const source = readFileSync(SOURCE_PATH, "utf8");

    expect(source).toContain("export function PremiumLoggingSettings");
    expect(source.match(SEMANTIC_TOKEN) ?? []).not.toHaveLength(0);
    expect(source.match(HARDCODED_PALETTE) ?? []).toHaveLength(0);
  });

  it("shows the enterprise notice and withholds the editor from a free user", () => {
    renderWithProviders(<PremiumLoggingSettings value={[]} onChange={vi.fn()} />);

    expect(screen.getByText(/LiteLLM Enterprise feature/)).toBeInTheDocument();
    expect(screen.getByText("✨ langfuse-logging")).toBeInTheDocument();
    expect(screen.queryByText("Logging Integrations")).not.toBeInTheDocument();
  });

  it("renders the editor for a premium user", () => {
    renderWithProviders(<PremiumLoggingSettings value={[]} onChange={vi.fn()} premiumUser />);

    expect(screen.getByText("Logging Integrations")).toBeInTheDocument();
    expect(screen.queryByText(/LiteLLM Enterprise feature/)).not.toBeInTheDocument();
  });
});
