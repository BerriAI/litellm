import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { teamSpendByUserCall } from "@/components/networking";

import TeamUserSpendCard from "./TeamUserSpendCard";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  teamSpendByUserCall: vi.fn(),
}));

const renderCard = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <TeamUserSpendCard
        accessToken="sk-test"
        startTime={new Date("2026-09-01T00:00:00Z")}
        endTime={new Date("2026-09-30T00:00:00Z")}
        teamIds={["team-alpha"]}
      />
    </QueryClientProvider>,
  );
};

describe("TeamUserSpendCard", () => {
  it("shows the server's error instead of an empty table when the request fails", async () => {
    const timeoutMessage = "Spend per user took longer than 30s. Pick a shorter date range or fewer teams";
    vi.mocked(teamSpendByUserCall).mockRejectedValueOnce(new Error(timeoutMessage));

    renderCard();

    expect(await screen.findByText(timeoutMessage)).toBeInTheDocument();
    expect(screen.queryByText("No user spend in this range")).not.toBeInTheDocument();
  });
});
