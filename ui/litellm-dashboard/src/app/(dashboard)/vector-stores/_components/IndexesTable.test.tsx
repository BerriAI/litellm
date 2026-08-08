import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { VectorStoreIndex } from "./IndexesTab";
import IndexesTable from "./IndexesTable";

const newerIndex: VectorStoreIndex = {
  id: "idx-newer",
  index_name: "newer-index",
  litellm_params: { vector_store_index: "provider-newer", vector_store_name: "newer-store" },
  created_by: "admin@example.com",
  created_at: "2024-02-20T10:30:00Z",
};

const olderIndex: VectorStoreIndex = {
  id: "idx-older",
  index_name: "older-index",
  litellm_params: { vector_store_index: "provider-older", vector_store_name: "older-store" },
  created_by: "admin@example.com",
  created_at: "2024-01-10T09:15:00Z",
};

const undatedIndex: VectorStoreIndex = {
  id: "idx-undated",
  index_name: "undated-index",
  litellm_params: { vector_store_index: "provider-undated", vector_store_name: "undated-store" },
  created_by: null,
  created_at: null,
};

describe("IndexesTable", () => {
  it("should display the empty state when no indexes are registered", () => {
    render(<IndexesTable data={[]} />);
    expect(screen.getByText("No indexes registered yet")).toBeInTheDocument();
  });

  it("should render index rows with dash fallbacks for missing created_by and created_at", () => {
    render(<IndexesTable data={[newerIndex, undatedIndex]} />);
    expect(screen.getByText("newer-index")).toBeInTheDocument();
    expect(screen.getByText("newer-store")).toBeInTheDocument();
    expect(screen.getByText("provider-newer")).toBeInTheDocument();
    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    const undatedRow = screen.getByText("undated-index").closest("tr");
    expect(undatedRow).not.toBeNull();
    expect(within(undatedRow as HTMLElement).getAllByText("-")).toHaveLength(2);
  });

  it("should sort by created_at descending by default", () => {
    render(<IndexesTable data={[olderIndex, newerIndex]} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-index")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-index")).toBeInTheDocument();
  });
});
