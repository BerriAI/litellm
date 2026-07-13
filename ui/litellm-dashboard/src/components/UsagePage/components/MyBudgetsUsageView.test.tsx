import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MyBudgetsUsageView from "./MyBudgetsUsageView";

vi.mock("../../team/MyUserTab", () => ({
  default: ({ teamId }: { teamId: string }) => <div data-testid="my-user-tab">MyUserTab:{teamId}</div>,
}));

vi.mock("antd", async () => {
  const React = await import("react");
  function Select(props: any) {
    const { value, onChange, options, ...rest } = props;
    return React.createElement(
      "select",
      {
        ...rest,
        role: "combobox",
        value: value ?? "",
        onChange: (e: any) => onChange?.(e.target.value),
      },
      (options || []).map((opt: any) =>
        React.createElement("option", { key: opt.value, value: opt.value }, opt.label),
      ),
    );
  }
  const Typography = {
    Text: ({ children, ...rest }: any) => React.createElement("span", rest, children),
  };
  return { Select, Typography };
});

describe("MyBudgetsUsageView", () => {
  it("shows empty state when user has no teams", () => {
    render(<MyBudgetsUsageView teams={[]} selectedTeamId={null} onTeamChange={vi.fn()} />);
    expect(screen.getByText(/not on any teams yet/i)).toBeInTheDocument();
  });

  it("uses selectedTeamId and renders MyUserTab", () => {
    render(
      <MyBudgetsUsageView
        teams={[
          { team_id: "team-a", team_alias: "Alpha" } as any,
          { team_id: "team-b", team_alias: "Beta" } as any,
        ]}
        selectedTeamId="team-b"
        onTeamChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("my-user-tab")).toHaveTextContent("MyUserTab:team-b");
  });

  it("defaults to the first team via onTeamChange when none selected", () => {
    const onTeamChange = vi.fn();
    render(
      <MyBudgetsUsageView
        teams={[
          { team_id: "team-a", team_alias: "Alpha" } as any,
          { team_id: "team-b", team_alias: "Beta" } as any,
        ]}
        selectedTeamId={null}
        onTeamChange={onTeamChange}
      />,
    );

    expect(onTeamChange).toHaveBeenCalledWith("team-a");
    expect(screen.getByTestId("my-user-tab")).toHaveTextContent("MyUserTab:team-a");
  });

  it("does not re-call onTeamChange when selectedTeamId is later cleared", () => {
    const onTeamChange = vi.fn();
    const teams = [
      { team_id: "team-a", team_alias: "Alpha" } as any,
      { team_id: "team-b", team_alias: "Beta" } as any,
    ];
    const { rerender } = render(
      <MyBudgetsUsageView teams={teams} selectedTeamId="team-a" onTeamChange={onTeamChange} />,
    );

    onTeamChange.mockClear();
    rerender(<MyBudgetsUsageView teams={teams} selectedTeamId={null} onTeamChange={onTeamChange} />);
    expect(onTeamChange).not.toHaveBeenCalled();
  });

  it("notifies parent when another team is selected", () => {
    const onTeamChange = vi.fn();
    render(
      <MyBudgetsUsageView
        teams={[
          { team_id: "team-a", team_alias: "Alpha" } as any,
          { team_id: "team-b", team_alias: "Beta" } as any,
        ]}
        selectedTeamId="team-a"
        onTeamChange={onTeamChange}
      />,
    );

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "team-b" } });
    expect(onTeamChange).toHaveBeenCalledWith("team-b");
  });
});
