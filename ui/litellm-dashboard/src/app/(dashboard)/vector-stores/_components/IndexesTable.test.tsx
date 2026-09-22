import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";

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
    renderWithProviders(<IndexesTable data={[]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />);
    expect(screen.getByText("No indexes registered yet")).toBeInTheDocument();
  });

  it("should render index rows with dash fallbacks for missing created_by and created_at", () => {
    renderWithProviders(
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
    renderWithProviders(
      <IndexesTable data={[olderIndex, newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-index")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-index")).toBeInTheDocument();
  });

  it("should call onViewVectorStore with the resolved id when the vector store cell is clicked", async () => {
    const user = userEvent.setup();
    const onViewVectorStore = vi.fn();
    renderWithProviders(
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
    renderWithProviders(
      <IndexesTable data={[newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    expect(screen.getByText("newer-store")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "newer-store" })).not.toBeInTheDocument();
  });

  it("should link created_by to the user detail deep link", () => {
    renderWithProviders(
      <IndexesTable data={[newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    const link = screen.getByRole("link", { name: "admin@example.com" });
    expect(link).toHaveAttribute("href", expect.stringMatching(/\/users\?user=admin%40example\.com$/));
  });

  it("should keep the dash fallback and render no link for a null created_by", () => {
    renderWithProviders(
      <IndexesTable data={[undatedIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
    );
    const row = screen.getByText("undated-index").closest("tr");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).queryByRole("link")).not.toBeInTheDocument();
    expect(within(row as HTMLElement).getAllByText("-").length).toBeGreaterThan(0);
  });

  describe("URL table state", () => {
    const manyIndexes: VectorStoreIndex[] = Array.from({ length: 30 }, (_, index) => ({
      id: `idx-${index}`,
      index_name: `index-${String(index).padStart(2, "0")}`,
      litellm_params: { vector_store_index: `provider-${index}`, vector_store_name: `store-${index}` },
      created_by: null,
      created_at: `2024-01-01T00:00:${String(59 - index).padStart(2, "0")}Z`,
    }));
    const rowNames = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getByText(/-index$|^index-/).textContent);
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];

    it("orders rows by the idx_ prefixed sort in the URL and ignores the unprefixed keys", () => {
      renderWithProviders(
        <IndexesTable data={[newerIndex, olderIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
        { searchParams: "?idx_sort_by=created_at&idx_sort_order=asc&sort_by=index_name&sort_order=desc" },
      );
      expect(rowNames()).toEqual(["older-index", "newer-index"]);
    });

    it.each(["index_name", "vector_store_name"])(
      "honors idx_sort_by=%s from the URL instead of falling back to created_at",
      (sortBy) => {
        const olderZulu: VectorStoreIndex = {
          ...olderIndex,
          id: "idx-zulu",
          index_name: "zulu-index",
          litellm_params: { vector_store_index: "provider-zulu", vector_store_name: "zulu-store" },
        };
        const newerAlpha: VectorStoreIndex = {
          ...newerIndex,
          id: "idx-alpha",
          index_name: "alpha-index",
          litellm_params: { vector_store_index: "provider-alpha", vector_store_name: "alpha-store" },
        };
        renderWithProviders(
          <IndexesTable data={[olderZulu, newerAlpha]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
          { searchParams: `?idx_sort_by=${sortBy}&idx_sort_order=asc` },
        );
        expect(rowNames()).toEqual(["alpha-index", "zulu-index"]);
      },
    );

    it("writes the sort under idx_ keys when a header is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const alphaIndex: VectorStoreIndex = { ...olderIndex, id: "idx-alpha", index_name: "alpha-index" };
      renderWithProviders(
        <IndexesTable data={[alphaIndex, newerIndex]} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
        { searchParams: "?sort_by=vector_store_id", onUrlUpdate },
      );
      expect(rowNames()).toEqual(["newer-index", "alpha-index"]);

      await user.click(screen.getByTestId("sort-header-index_name"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("idx_sort_by")).toBe("index_name"));
      const params = lastUrlUpdate(onUrlUpdate)?.searchParams;
      expect(params?.get("idx_sort_order")).toBe("asc");
      expect(params?.get("sort_by")).toBe("vector_store_id");
      expect(rowNames()).toEqual(["alpha-index", "newer-index"]);
    });

    it("opens the page named by idx_page", () => {
      renderWithProviders(
        <IndexesTable data={manyIndexes} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
        { searchParams: "?idx_page=2&page=1" },
      );
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
      expect(rowNames()).toEqual(["index-25", "index-26", "index-27", "index-28", "index-29"]);
    });

    it("writes idx_page when paging forward", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(
        <IndexesTable data={manyIndexes} resolveVectorStoreId={noResolve} onViewVectorStore={vi.fn()} />,
        { onUrlUpdate },
      );

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("idx_page")).toBe("2"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
      expect(rowNames()[0]).toBe("index-25");
    });
  });
});
