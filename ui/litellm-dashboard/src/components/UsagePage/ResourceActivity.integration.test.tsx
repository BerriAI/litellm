import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
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
    expect(networking.dailyActivityAggregatedCall).toHaveBeenCalledWith(
      entity,
      expect.objectContaining({ ...filter, startTime: expect.any(Date), endTime: expect.any(Date) }),
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
