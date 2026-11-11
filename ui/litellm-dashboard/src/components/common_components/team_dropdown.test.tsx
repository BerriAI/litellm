import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { chooseSelectOption } from "../../../tests/test-utils";
import type { Team } from "../key_team_helpers/key_list";
import TeamDropdown from "./team_dropdown";

const TEAMS = [
  { team_id: "team-1", team_alias: "Alpha Team" },
  { team_id: "team-2", team_alias: "Beta Team" },
] as unknown as Team[];

const mockUseInfiniteTeams = vi.hoisted(() => vi.fn());
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useInfiniteTeams: mockUseInfiniteTeams }));
const teamQuery = {
  data: { pages: [{ teams: TEAMS }] },
  fetchNextPage: vi.fn(),
  hasNextPage: false,
  isFetchingNextPage: false,
  isLoading: false,
};

describe("TeamDropdown", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteTeams.mockReturnValue(teamQuery);
  });

  it("loads past unauthorized teams so a permitted team on the next page can be selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const filterTeam = (team: Team) => team.team_id === "team-2";
    mockUseInfiniteTeams.mockReturnValue({
      ...teamQuery,
      data: { pages: [{ teams: [TEAMS[0]] }] },
      hasNextPage: true,
    });
    const view = render(<TeamDropdown filterTeam={filterTeam} onChange={onChange} />);
    await waitFor(() => expect(teamQuery.fetchNextPage).toHaveBeenCalledOnce());
    mockUseInfiniteTeams.mockReturnValue(teamQuery);
    view.rerender(<TeamDropdown filterTeam={filterTeam} onChange={onChange} />);
    await user.click(screen.getByRole("combobox"));
    expect(screen.queryByRole("option", { name: /Alpha Team/ })).not.toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /Beta Team/ }));
    expect(onChange).toHaveBeenCalledWith("team-2");
    expect(teamQuery.fetchNextPage).toHaveBeenCalledOnce();
  });

  it("emits the picked team's id and full object", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onTeamSelect = vi.fn();
    render(<TeamDropdown onChange={onChange} onTeamSelect={onTeamSelect} />);

    await chooseSelectOption(user, screen.getByRole("combobox"), /^Beta Team/);

    expect(onChange).toHaveBeenCalledWith("team-2");
    expect(onTeamSelect).toHaveBeenCalledWith(TEAMS[1]);
  });

  it("emits null, never the empty string, when the selection is cleared", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onTeamSelect = vi.fn();
    render(<TeamDropdown value="team-1" onChange={onChange} onTeamSelect={onTeamSelect} />);

    await user.click(screen.getByRole("button", { name: "Clear" }));

    expect(onChange).toHaveBeenCalledWith(null);
    expect(onTeamSelect).toHaveBeenCalledWith(null);
  });
});
