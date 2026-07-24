import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { MemoryRow } from "@/components/networking";

import { MemoryView } from "./MemoryView";

interface CapturedTableProps {
  isLoading: boolean;
  rowCount: number;
  data: MemoryRow[];
  hasActiveSearch: boolean;
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
});
