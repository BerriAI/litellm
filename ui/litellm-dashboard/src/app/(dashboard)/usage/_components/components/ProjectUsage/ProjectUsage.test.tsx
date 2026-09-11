import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";
import * as networking from "@/components/networking";

import ProjectUsage from "./ProjectUsage";
import type { EntityList } from "../EntityUsage/EntityUsage";

vi.mock("@/components/networking", () => ({
  projectDailyActivityCall: vi.fn(),
}));

const PROJECT_LIST: EntityList[] = [
  { label: "Project Alpha", value: "project-alpha" },
  { label: "Project Beta", value: "project-beta" },
];

const DATE_VALUE = { from: new Date("2026-09-01"), to: new Date("2026-09-08") };

describe("ProjectUsage", () => {
  const mockProjectDailyActivityCall = vi.mocked(networking.projectDailyActivityCall);

  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
  });

  it("shows an enterprise upsell and never calls the API when the caller is not a premium user", () => {
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={false} />,
    );

    expect(screen.getByText("Project Usage is an Enterprise feature")).toBeInTheDocument();
    expect(mockProjectDailyActivityCall).not.toHaveBeenCalled();
  });

  it("prompts for a project before fetching anything", () => {
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={true} />,
    );

    expect(screen.getByText("Select at least one project above to view its usage.")).toBeInTheDocument();
    expect(mockProjectDailyActivityCall).not.toHaveBeenCalled();
  });

  it("fetches and renders the picked project's daily spend", async () => {
    mockProjectDailyActivityCall.mockResolvedValue({
      start_date: "2026-09-01",
      end_date: "2026-09-08",
      results: [
        {
          date: "2026-09-01",
          project_id: "project-alpha",
          project_alias: "Project Alpha",
          spend: 12.5,
          prompt_tokens: 100,
          completion_tokens: 50,
          total_tokens: 150,
          api_requests: 4,
          successful_requests: 3,
          failed_requests: 1,
        },
      ],
    });

    const user = userEvent.setup();
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={true} />,
    );

    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Project Alpha" }));

    await waitFor(() => expect(mockProjectDailyActivityCall).toHaveBeenCalledTimes(1));
    expect(mockProjectDailyActivityCall).toHaveBeenCalledWith("test-token", DATE_VALUE.from, DATE_VALUE.to, [
      "project-alpha",
    ]);

    await waitFor(() => expect(screen.getAllByText("$12.50").length).toBeGreaterThan(0));
    expect(screen.getAllByText("4").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Project Alpha").length).toBeGreaterThan(0);
  });

  it("shows an error instead of silently rendering zeros when the request fails", async () => {
    mockProjectDailyActivityCall.mockRejectedValue(new Error("Project management is an enterprise feature"));

    const user = userEvent.setup();
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={true} />,
    );

    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Project Alpha" }));

    expect(await screen.findByText("Could not load project usage")).toBeInTheDocument();
    expect(screen.getByText("Project management is an enterprise feature")).toBeInTheDocument();
    expect(screen.queryByText("No project usage data")).not.toBeInTheDocument();
  });

  it("renders a backend not-found message verbatim, without mangling the comma-joined list", async () => {
    mockProjectDailyActivityCall.mockRejectedValue(new Error("Project(s) not found: also-missing, does-not-exist"));

    const user = userEvent.setup();
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={true} />,
    );

    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Project Alpha" }));

    expect(await screen.findByText("Project(s) not found: also-missing, does-not-exist")).toBeInTheDocument();
  });

  it("shows a loading indicator while fetching data for a newly-added project", async () => {
    let resolveSecondCall: (value: unknown) => void = () => {};
    mockProjectDailyActivityCall
      .mockResolvedValueOnce({
        start_date: "2026-09-01",
        end_date: "2026-09-08",
        results: [
          {
            date: "2026-09-01",
            project_id: "project-alpha",
            project_alias: "Project Alpha",
            spend: 12.5,
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            api_requests: 4,
            successful_requests: 3,
            failed_requests: 1,
          },
        ],
      })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecondCall = resolve;
          }),
      );

    const user = userEvent.setup();
    renderWithProviders(
      <ProjectUsage accessToken="test-token" projectList={PROJECT_LIST} dateValue={DATE_VALUE} premiumUser={true} />,
    );

    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Project Alpha" }));
    await waitFor(() => expect(screen.getAllByText("$12.50").length).toBeGreaterThan(0));

    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: "Project Beta" }));

    await waitFor(() => expect(screen.getAllByText("Loading chart data...").length).toBeGreaterThan(0));

    resolveSecondCall({
      start_date: "2026-09-01",
      end_date: "2026-09-08",
      results: [],
    });

    await waitFor(() => expect(screen.queryAllByText("Loading chart data...").length).toBe(0));
  });
});
