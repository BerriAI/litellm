import { act, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { Team } from "@/components/key_team_helpers/key_list";
import ModelsCell from "./ModelsCell";

// The Icon component from @tremor/react does not forward onClick to the rendered element
// by default in the test environment, so we stub it with a clickable button so accordion
// interaction can be tested end-to-end.
vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Icon: ({ onClick, "aria-label": ariaLabel }: { onClick?: () => void; "aria-label"?: string }) =>
      React.createElement("button", { onClick, "aria-label": ariaLabel ?? "accordion-toggle", type: "button" }),
  };
});

const makeTeam = (models: string[], overrides: Partial<Team> = {}): Team => ({
  team_id: "team-1",
  team_alias: "Engineering",
  models,
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "org-1",
  created_at: "2024-01-01T00:00:00Z",
  keys: [],
  members_with_roles: [],
  spend: 0,
  ...overrides,
});

// Wrap in a table so the <td> from TableCell renders without HTML warnings.
const renderModelsCell = (team: Team) =>
  render(
    <table>
      <tbody>
        <tr>
          <ModelsCell team={team} />
        </tr>
      </tbody>
    </table>,
  );

describe("ModelsCell", () => {
  it("should show 'All Proxy Models' badge when the models array is empty", () => {
    renderModelsCell(makeTeam([]));

    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });

  it("should show an 'All Proxy Models' badge when the model value is 'all-proxy-models'", () => {
    renderModelsCell(makeTeam(["all-proxy-models"]));

    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });

  it("should display individual model badges for up to 3 models without an accordion", () => {
    renderModelsCell(makeTeam(["gpt-4", "gpt-3.5-turbo", "claude-3"]));

    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
    expect(screen.getByText("claude-3")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /accordion/i })).not.toBeInTheDocument();
  });

  it("should truncate model names longer than 30 characters with an ellipsis", () => {
    const longName = "a-very-long-model-name-exceeding-thirty-chars";
    renderModelsCell(makeTeam([longName]));

    const badge = screen.getByText((text) => text.endsWith("..."));
    expect(badge).toBeInTheDocument();
    expect(badge.textContent!.length).toBeLessThanOrEqual(33); // 30 chars + "..."
  });

  it("should show the first 3 models and a '+N more models' badge when there are more than 3 models", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "m4", "m5"]));

    expect(screen.getByText("m1")).toBeInTheDocument();
    expect(screen.getByText("m2")).toBeInTheDocument();
    expect(screen.getByText("m3")).toBeInTheDocument();
    expect(screen.getByText("+2 more models")).toBeInTheDocument();
    expect(screen.queryByText("m4")).not.toBeInTheDocument();
    expect(screen.queryByText("m5")).not.toBeInTheDocument();
  });

  it("should use singular 'more model' when there is exactly 1 overflow model", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "m4"]));

    expect(screen.getByText("+1 more model")).toBeInTheDocument();
  });

  it("should show the accordion toggle button when there are more than 3 models", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "m4"]));

    expect(screen.getByRole("button", { name: /accordion/i })).toBeInTheDocument();
  });

  it("should expand to show all models when the accordion toggle is clicked", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "m4", "m5"]));

    act(() => {
      screen.getByRole("button", { name: /accordion/i }).click();
    });

    expect(screen.getByText("m4")).toBeInTheDocument();
    expect(screen.getByText("m5")).toBeInTheDocument();
    expect(screen.queryByText("+2 more models")).not.toBeInTheDocument();
  });

  it("should collapse back to show the overflow badge after a second click on the toggle", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "m4", "m5"]));

    const toggle = screen.getByRole("button", { name: /accordion/i });
    act(() => {
      toggle.click();
    });
    act(() => {
      toggle.click();
    });

    expect(screen.queryByText("m4")).not.toBeInTheDocument();
    expect(screen.getByText("+2 more models")).toBeInTheDocument();
  });

  it("should render 'all-proxy-models' entries in the overflow section as 'All Proxy Models' badges", () => {
    renderModelsCell(makeTeam(["m1", "m2", "m3", "all-proxy-models"]));

    act(() => {
      screen.getByRole("button", { name: /accordion/i }).click();
    });

    // There should now be an "All Proxy Models" badge in the expanded section
    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
  });
});
