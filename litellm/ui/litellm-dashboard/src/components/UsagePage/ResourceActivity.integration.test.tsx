import { renderWithProviders as render } from "../../../tests/test-utils";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { useActivityDateRange } from "@/app/(dashboard)/cost-optimization/_components/useDailyActivityRange";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import * as networking from "@/components/networking";
import { EMPTY_DAILY_ACTIVITY_METADATA, type DailyActivityAggregatedResponse } from "./dailyActivityApi";
import ResourceActivity from "./ResourceActivity";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: vi.fn() }));

const auth = {
  accessToken: "test-session",
  userId: "user-1",
  userRole: "Admin",
} as ReturnType<typeof useAuthorized>;

const response: DailyActivityAggregatedResponse = {
  results: [
    {
      date: "2026-11-01",
      metrics: {
        spend: 12.34,
        api_requests: 42,
        successful_requests: 40,
        failed_requests: 2,
        total_tokens: 4200,
        prompt_tokens: 3000,
        completion_tokens: 1200,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      breakdown: {},
    },
  ],
  metadata: {
    ...EMPTY_DAILY_ACTIVITY_METADATA,
    total_spend: 12.34,
    total_api_requests: 42,
    total_successful_requests: 40,
    total_failed_requests: 2,
    total_tokens: 4200,
  },
};

describe("ResourceActivity", () => {
  afterEach(() => vi.useRealTimers());

  it("keeps alert dates when Activity mounts after another date consumer", async () => {
    function Details() {
      useActivityDateRange();
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>Open Activity</button>
          {open && <ResourceActivity scope="team" id="resource-1" />}
        </>
      );
    }
    render(<Details />, { searchParams: "activity_from=2026-01-01&activity_to=2026-01-08" });
    fireEvent.click(screen.getByRole("button", { name: "Open Activity" }));
    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      "team",
      expect.objectContaining({ startTime: new Date(2026, 0, 1, 12), endTime: new Date(2026, 0, 8, 12) }),
    );
  });

  it("preserves an unfinished date edit when the parent rerenders", async () => {
    const { rerender } = render(<ResourceActivity scope="team" id="resource-1" />, {
      searchParams: "activity_from=2026-01-01&activity_to=2026-01-08",
    });
    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "1 Jan, 12:00 - 8 Jan, 12:00" }));
    fireEvent.change(screen.getByLabelText("Start date"), { target: { value: "2026-01-02" } });
    rerender(<ResourceActivity scope="team" id="resource-1" />);
    expect(screen.getByLabelText("Start date")).toHaveValue("2026-01-02");
  });

  it.each(["team", "user"] as const)("shows the returned model breakdown for a %s", async (scope) => {
    vi.mocked(networking.dailyActivityAggregatedCall).mockResolvedValue({
      ...response,
      results: [
        {
          ...response.results[0],
          breakdown: {
            models: {
              "example-model": { metrics: response.results[0].metrics },
            },
          },
        },
      ],
    });
    render(<ResourceActivity scope={scope} id="resource-1" />);
    expect(await screen.findByRole("cell", { name: "example-model" })).toBeInTheDocument();
  });

  it("shows matching recorded models and their totals for a wildcard, in one request", async () => {
    vi.mocked(networking.dailyActivityAggregatedCall).mockResolvedValue({
      ...response,
      metadata: { ...response.metadata, total_spend: 124.68 },
      results: [
        {
          ...response.results[0],
          breakdown: {
            models: {
              "openai/model-one": { metrics: response.results[0].metrics },
              "openai/model-two": { metrics: response.results[0].metrics },
              "anthropic/other-model": { metrics: { ...response.results[0].metrics, spend: 100 } },
            },
          },
        },
      ],
    });
    render(<ResourceActivity scope="model" id="openai/*" />);
    expect(await screen.findByText("$24.68")).toBeInTheDocument();
    expect(screen.getByText("openai/model-one")).toBeInTheDocument();
    expect(screen.getByText("openai/model-two")).toBeInTheDocument();
    expect(screen.queryByText("anthropic/other-model")).not.toBeInTheDocument();
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledExactlyOnceWith(
      "user",
      expect.objectContaining({ model: undefined, includeEntityBreakdown: false }),
    );
  });

  it("keeps wildcard usage within the viewer's scope and shows empty when no model matches", async () => {
    vi.mocked(useAuthorized).mockReturnValue({ ...auth, userRole: "Internal User" });
    render(<ResourceActivity scope="model" id="openai/*" />);
    expect(await screen.findByText("No daily details available")).toBeInTheDocument();
    expect(screen.queryByText("$12.34")).not.toBeInTheDocument();
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      "user",
      expect.objectContaining({ model: undefined, entityIds: [auth.userId] }),
    );
  });

  it("includes today's UTC bucket for a team when the local date is still yesterday", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-09T01:00:00Z"));
    vi.spyOn(networking, "formatDate").mockImplementation((date) =>
      new Date(date.getTime() - 8 * 60 * 60 * 1000).toISOString().slice(0, 10),
    );
    render(<ResourceActivity scope="team" id="team-1" />);
    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    const request = vi.mocked(networking.dailyActivityAggregatedCall).mock.calls[0][1];
    expect(request.endTime.getFullYear()).toBe(2026);
    expect(request.endTime.getMonth()).toBe(10);
    expect(request.endTime.getDate()).toBe(9);
  });

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.mocked(useAuthorized).mockReturnValue(auth);
    vi.spyOn(networking, "dailyActivityAggregatedCall").mockResolvedValue(response);
  });

  it.each([
    { scope: "key", id: "key-1", entity: "user", filter: { apiKey: "key-1", entityIds: null } },
    { scope: "team", id: "team-1", entity: "team", filter: { entityIds: ["team-1"] } },
    { scope: "user", id: "user-2", entity: "user", filter: { entityIds: ["user-2"] } },
    { scope: "model", id: "example-model", entity: "user", filter: { model: "example-model", entityIds: null } },
  ] as const)("shows existing Activity metrics filtered to the $scope", async ({ scope, id, entity, filter }) => {
    render(<ResourceActivity scope={scope} id={id} />);

    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    const expectedRequest = {
      ...filter,
      includeEntityBreakdown: scope === "model" ? false : undefined,
      startTime: expect.any(Date),
      endTime: expect.any(Date),
    };
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      entity,
      expect.objectContaining(expectedRequest),
    );
  });

  it("keeps a non-admin key view within that user's spend scope", async () => {
    vi.mocked(useAuthorized).mockReturnValue({ ...auth, userRole: "Internal User" });
    render(<ResourceActivity scope="key" id="key-1" />);

    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    expect(screen.getByText("Showing your usage only.")).toBeInTheDocument();
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      "user",
      expect.objectContaining({ entityIds: ["user-1"], apiKey: "key-1" }),
    );
  });

  it("explains that team Activity is limited to the keys the viewer can access", async () => {
    vi.mocked(useAuthorized).mockReturnValue({ ...auth, userRole: "Internal User" });
    render(<ResourceActivity scope="team" id="team-1" />);

    expect(await screen.findByText("$12.34")).toBeInTheDocument();
    expect(screen.getByText("Showing usage for the team keys you have permission to view.")).toBeInTheDocument();
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      "team",
      expect.objectContaining({ entityIds: ["team-1"] }),
    );
  });

  it("does not query another user's spend for a non-admin", () => {
    vi.mocked(useAuthorized).mockReturnValue({ ...auth, userRole: "Internal User" });
    render(<ResourceActivity scope="user" id="user-2" />);

    expect(networking.dailyActivityAggregatedCall).not.toHaveBeenCalled();
    expect(screen.queryByText("$12.34")).not.toBeInTheDocument();
  });

  it("shows a retryable error instead of zero usage when the request fails", async () => {
    vi.mocked(networking.dailyActivityAggregatedCall).mockRejectedValueOnce(new Error("Activity unavailable"));
    render(<ResourceActivity scope="team" id="team-1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Activity unavailable");
    expect(screen.queryByText("$12.34")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(await screen.findByText("$12.34")).toBeInTheDocument();
  });

  it("replaces the previous resource's data when navigating to another resource", async () => {
    const { rerender } = render(<ResourceActivity scope="team" id="team-1" />);
    expect(await screen.findByText("$12.34")).toBeInTheDocument();

    vi.mocked(networking.dailyActivityAggregatedCall).mockResolvedValue({
      results: [],
      metadata: EMPTY_DAILY_ACTIVITY_METADATA,
    });
    rerender(<ResourceActivity scope="team" id="team-2" />);

    expect(await screen.findByText("No daily details available")).toBeInTheDocument();
    expect(screen.queryByText("$12.34")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(networking.dailyActivityAggregatedCall).toHaveBeenLastCalledWith(
        "team",
        expect.objectContaining({ entityIds: ["team-2"] }),
      ),
    );
  });
});
