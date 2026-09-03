import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RouterSettingsSummary from "./RouterSettingsSummary";

describe("RouterSettingsSummary", () => {
  it("should list each configured fallback mapping", () => {
    render(
      <RouterSettingsSummary
        routerSettings={{
          fallbacks: [{ "gpt-4": ["gpt-4o", "claude-sonnet"] }, { "gpt-4o": ["gpt-4o-mini"] }],
          num_retries: 3,
        }}
      />,
    );

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o, claude-sonnet")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o-mini")).toBeInTheDocument();
    expect(screen.getByText("Number of Retries: 3")).toBeInTheDocument();
  });

  it("should show the empty state when every setting is null", () => {
    render(<RouterSettingsSummary routerSettings={{ fallbacks: null, num_retries: null }} />);

    expect(screen.getByText("No router settings configured")).toBeInTheDocument();
  });
});
