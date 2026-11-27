import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { userConnectionsListCall } from "@/components/networking";

import UserConnectionsPanel from "./UserConnectionsPanel";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "token", userId: "u", userRole: "internal_user" }),
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    userConnectionsListCall: vi.fn(),
    userConnectionDeleteCall: vi.fn(),
  };
});

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <UserConnectionsPanel />
    </QueryClientProvider>,
  );
}

describe("UserConnectionsPanel", () => {
  it("shows an error state with retry instead of an empty list when the query fails", async () => {
    vi.mocked(userConnectionsListCall).mockRejectedValue(new Error("boom"));
    renderPanel();
    expect(await screen.findByText("Could not load your connections.")).toBeInTheDocument();
    expect(screen.queryByText("No credentials need a personal connection.")).not.toBeInTheDocument();
  });

  it("refetches when retry is clicked", async () => {
    const user = userEvent.setup();
    vi.mocked(userConnectionsListCall)
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce({ connections: [] });
    renderPanel();
    await user.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findByText("No credentials need a personal connection.")).toBeInTheDocument();
    expect(vi.mocked(userConnectionsListCall).mock.calls.length).toBeGreaterThanOrEqual(2);
  });
});
