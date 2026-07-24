import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { MemoryRow } from "@/components/networking";

import { MemoryView } from "./MemoryView";

interface CapturedTableProps {
  isLoading: boolean;
  rowCount: number;
  data: MemoryRow[];
  hasActiveSearch: boolean;
  onViewClick: (row: MemoryRow) => void;
}

const captured = vi.hoisted(() => ({ current: null as CapturedTableProps | null }));

vi.mock("./MemoryTable", () => ({
  MemoryTable: function MemoryTableMock(props: CapturedTableProps) {
    captured.current = props;
    return <div data-testid="memory-table-mock" />;
  },
}));

const renderView = (accessToken: string | null) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryView accessToken={accessToken} userID={null} userRole={null} />
    </QueryClientProvider>,
  );
};

describe("MemoryView", () => {
  it("keeps the table out of the skeleton state when the token is null (disabled query)", () => {
    renderView(null);

    expect(captured.current).not.toBeNull();
    expect(captured.current?.isLoading).toBe(false);
    expect(captured.current?.data).toEqual([]);
    expect(captured.current?.rowCount).toBe(0);
    expect(captured.current?.hasActiveSearch).toBe(false);
  });

  it("heads the page with the Memory title and the /v1/memory scope note", () => {
    renderView(null);

    expect(screen.getByRole("heading", { name: "Memory" })).toBeInTheDocument();
    expect(screen.getByText("/v1/memory")).toBeInTheDocument();
    expect(screen.getByText(/Scoped to memories visible to your user \/ team \(admins see all\)/)).toBeInTheDocument();
  });

  it("opens the create modal from the New memory button", async () => {
    const user = userEvent.setup();
    renderView(null);

    expect(screen.queryByText("Create memory")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /new memory/i }));

    expect(await screen.findByText("Create memory")).toBeInTheDocument();
  });

  it("opens the detail drawer for the row the table hands back, and closes it again", async () => {
    const user = userEvent.setup();
    renderView(null);

    expect(screen.queryByText("Memory ID")).not.toBeInTheDocument();

    const row: MemoryRow = {
      memory_id: "mem-drawer",
      key: "user:profile",
      value: "remembered",
      metadata: null,
      user_id: null,
      team_id: null,
    };
    act(() => captured.current?.onViewClick(row));

    expect(await screen.findByText("Memory ID")).toBeInTheDocument();
    expect(screen.getByText("mem-drawer")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(screen.queryByText("mem-drawer")).not.toBeInTheDocument();
  });
});
