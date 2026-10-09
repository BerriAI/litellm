import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { BuilderInsightBuilder } from "./builderInsightsData";
import { BuilderList } from "./BuilderList";

vi.mock("../overview/Primitives", () => ({
  Panel: ({ title, children }: { title: string; children: ReactNode }) => (
    <section>
      <h2>{title}</h2>
      {children}
    </section>
  ),
  Segmented: () => null,
}));

vi.mock("../overview/TopAgents", () => ({
  AgentMark: () => null,
}));

const builder: BuilderInsightBuilder = {
  id: "sample-builder",
  name: "Sample Builder",
  email: "sample@example.test",
  verdict: "productive",
  verdictLabel: "Productive",
  verdictLine: "Ships work productively",
  archetype: "Routing Architect",
  tagline: "Routes requests through `codex` workflows",
  uses: "",
  markdown: "",
  spend: 1_200,
  requests: 1,
  tokens: 0,
  cacheHitRate: 0,
  failed: 0,
  activeDays: 1,
  prsOwn: 1,
  prsDevin: 0,
  prs: 1,
  spendPerPr: 32_600,
  medianPromptTokens: 0,
  p90PromptTokens: 0,
  daily: [],
  models: [],
  agents: [],
  hoursPdt: [],
  heatPdt: [],
  topSessions: [],
  prScopes: [],
};

describe("BuilderList", () => {
  it("shows names and archetypes in rows and exposes the full tagline without backticks in the title", () => {
    render(
      <BuilderList
        builders={[builder]}
        selectedId={builder.id}
        sort="spend"
        onSelect={() => {}}
        onSortChange={() => {}}
      />,
    );

    expect(screen.getByText("Sample Builder")).toBeInTheDocument();
    expect(screen.getByText("Productive")).toBeInTheDocument();
    expect(screen.queryByText(/Routes requests through/)).not.toBeInTheDocument();
    expect(screen.getByRole("button")).toHaveAttribute(
      "title",
      "Routes requests through codex workflows",
    );
  });
});
