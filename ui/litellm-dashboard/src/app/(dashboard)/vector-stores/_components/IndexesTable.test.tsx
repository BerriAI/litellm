import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { VectorStoreIndex } from "./IndexesTab";
import IndexesTable from "./IndexesTable";

vi.mock("next/navigation", async () => ({
  ...(await vi.importActual("next/navigation")),
  useRouter: () => ({ push: vi.fn() }),
}));

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

const noResolve = () => undefined;

describe("IndexesTable", () => {
  it("should display the empty state when no indexes are registered", () => {
    render(<IndexesTable data={[]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />);
    expect(screen.getByText("No indexes registered yet")).toBeInTheDocument();
  });

  it("should render index rows with dash fallbacks for missing created_by and created_at", () => {
    render(
      <IndexesTable data={[newerIndex, undatedIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    expect(screen.getByText("newer-index")).toBeInTheDocument();
    expect(screen.getByText("newer-store")).toBeInTheDocument();
    expect(screen.getByText("provider-newer")).toBeInTheDocument();
    expect(screen.getByText("admin@example.com")).toBeInTheDocument();
    const undatedRow = screen.getByText("undated-index").closest("tr");
    expect(undatedRow).not.toBeNull();
    expect(within(undatedRow as HTMLElement).getAllByText("-")).toHaveLength(2);
  });

  it("should sort by created_at descending by default", () => {
    render(
      <IndexesTable data={[olderIndex, newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-index")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-index")).toBeInTheDocument();
  });

  it("should call onViewVectorStore with the resolved id when the vector store cell is clicked", async () => {
    const user = userEvent.setup();
    const onViewVectorStore = vi.fn();
    render(
      <IndexesTable
        data={[newerIndex]}
        resolveVectorStoreId={(name) => (name === "newer-store" ? "vs-newer" : undefined)}
        onViewVectorStore={onViewVectorStore}
      />,
    );
    await user.click(screen.getByRole("button", { name: "newer-store" }));
    expect(onViewVectorStore).toHaveBeenCalledWith("vs-newer");
  });

  it("should render an unresolvable vector store name as plain text without a clickable cell", () => {
    render(<IndexesTable data={[newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />);
    expect(screen.getByText("newer-store")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "newer-store" })).not.toBeInTheDocument();
  });

  it("should link created_by to the user detail deep link", () => {
    render(<IndexesTable data={[newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />);
    const link = screen.getByRole("link", { name: "admin@example.com" });
    expect(link).toHaveAttribute("href", expect.stringMatching(/\/users\?user=admin%40example\.com$/));
  });

  it("should keep the dash fallback and render no link for a null created_by", () => {
    render(<IndexesTable data={[undatedIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />);
    const row = screen.getByText("undated-index").closest("tr");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).queryByRole("link")).not.toBeInTheDocument();
    expect(within(row as HTMLElement).getAllByText("-").length).toBeGreaterThan(0);
  });
});
