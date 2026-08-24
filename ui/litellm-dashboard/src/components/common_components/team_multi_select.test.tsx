import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useInfiniteTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import TeamMultiSelect from "./team_multi_select";

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: vi.fn(),
}));

const team = (id: string, alias: string) => ({ team_id: id, team_alias: alias });

const mockTeamsResult = (
  overrides: Partial<{
    pages: { teams: ReturnType<typeof team>[] }[];
    isLoading: boolean;
    hasNextPage: boolean;
    isFetchingNextPage: boolean;
  }> = {},
) => {
  const { pages = [{ teams: [team("team-1", "Alpha Team"), team("team-2", "Beta Team")] }], ...rest } = overrides;
  return {
    data: { pages },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
    ...rest,
  };
};

describe("TeamMultiSelect", () => {
  const mockUseInfiniteTeams = vi.mocked(useInfiniteTeams);

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult() as never);
  });

  const combobox = () => screen.getByRole("combobox");

  // One library paints the prompt as its own text node and the other leaves it on the input's
  // placeholder attribute, so either one means the user is being told what to type.
  const promptsWith = (text: string) =>
    screen.queryAllByText(text).length + screen.queryAllByPlaceholderText(text).length > 0;

  it("renders a search control with the given placeholder", () => {
    render(<TeamMultiSelect placeholder="Search teams by alias..." />);

    expect(combobox()).toBeInTheDocument();
    expect(promptsWith("Search teams by alias...")).toBe(true);
  });

  it("offers every loaded team by alias and id", async () => {
    const user = userEvent.setup();
    render(<TeamMultiSelect />);

    await user.click(combobox());

    expect(screen.getByText("Alpha Team")).toBeInTheDocument();
    expect(screen.getByText("(team-1)")).toBeInTheDocument();
    expect(screen.getByText("Beta Team")).toBeInTheDocument();
    expect(screen.getByText("(team-2)")).toBeInTheDocument();
  });

  it("deduplicates a team that appears on more than one page", async () => {
    mockUseInfiniteTeams.mockReturnValue(
      mockTeamsResult({
        pages: [
          { teams: [team("team-1", "Alpha Team")] },
          { teams: [team("team-1", "Alpha Team"), team("team-2", "Beta Team")] },
        ],
      }) as never,
    );
    const user = userEvent.setup();
    render(<TeamMultiSelect />);

    await user.click(combobox());

    expect(screen.getAllByText("Alpha Team")).toHaveLength(1);
    expect(screen.getByText("Beta Team")).toBeInTheDocument();
  });

  it("reports the picked team id to onChange", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<TeamMultiSelect onChange={onChange} />);

    await user.click(combobox());
    const matches = screen.getAllByText("Beta Team");
    await user.click(matches[matches.length - 1]);

    expect(onChange).toHaveBeenCalled();
    expect(onChange.mock.calls[0][0]).toEqual(["team-2"]);
  });

  it("does not report a selection while disabled", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<TeamMultiSelect onChange={onChange} disabled />);

    await user.click(combobox());

    expect(screen.queryByText("Alpha Team")).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("tells the user when there are no teams to pick", async () => {
    mockUseInfiniteTeams.mockReturnValue(mockTeamsResult({ pages: [{ teams: [] }] }) as never);
    const user = userEvent.setup();
    render(<TeamMultiSelect />);

    await user.click(combobox());

    // Substring match because one library appends an invisible word joiner for its live region.
    expect(screen.getByText(/No teams found/)).toBeInTheDocument();
  });

  it("passes the page size and organization filter through to the teams query", () => {
    render(<TeamMultiSelect pageSize={25} organizationId="org-7" />);

    expect(mockUseInfiniteTeams).toHaveBeenCalledWith(25, undefined, "org-7");
  });
});
