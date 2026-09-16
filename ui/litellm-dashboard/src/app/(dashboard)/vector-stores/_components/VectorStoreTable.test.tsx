import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { VectorStore } from "@/components/vector_store_management/types";

import VectorStoreTable from "./VectorStoreTable";

vi.mock("@/components/vector_store_providers", () => ({
  getVectorStoreProviderLogoAndName: (provider: string) => {
    const providerMap: Record<string, { displayName: string; logo: string }> = {
      openai: { displayName: "OpenAI", logo: "/openai-logo.png" },
      azure: { displayName: "Azure", logo: "/azure-logo.png" },
    };
    return providerMap[provider] || { displayName: provider, logo: "" };
  },
}));

const mockVectorStores: VectorStore[] = [
  {
    vector_store_id: "vs-newer",
    custom_llm_provider: "openai",
    vector_store_name: "My OpenAI Store",
    vector_store_description: "A store for OpenAI vectors",
    vector_store_metadata: {
      ingested_files: [
        { filename: "a.pdf", ingested_at: "2024-01-15T10:00:00Z" },
        { filename: "b.pdf", ingested_at: "2024-01-15T10:00:00Z" },
      ],
    },
    created_at: "2024-01-15T10:30:00Z",
    updated_at: "2024-01-15T11:00:00Z",
  },
  {
    vector_store_id: "vs-older",
    custom_llm_provider: "azure",
    vector_store_name: undefined,
    vector_store_description: undefined,
    created_at: "2024-01-10T09:15:00Z",
    updated_at: "2024-01-12T14:20:00Z",
  },
];

const mockOnView = vi.fn();
const mockOnEdit = vi.fn();
const mockOnDelete = vi.fn();

const defaultProps = {
  data: mockVectorStores,
  onView: mockOnView,
  onEdit: mockOnEdit,
  onDelete: mockOnDelete,
};

describe("VectorStoreTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    for (const header of ["Vector Store ID", "Name", "Description", "Files", "Provider", "Created At", "Updated At"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display the empty state when data is empty", () => {
    renderWithProviders(<VectorStoreTable {...defaultProps} data={[]} />);
    expect(screen.getByText("No vector stores")).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("vs-newer")).toBeInTheDocument();
    expect(within(rows[1]).getByText("vs-older")).toBeInTheDocument();
  });

  it("should call onView when the vector store ID is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "vs-newer" }));
    expect(mockOnView).toHaveBeenCalledWith("vs-newer");
  });

  it("should render provider display names", () => {
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
  });

  it("should summarize ingested files and fall back to a dash without files", () => {
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    expect(screen.getByText("2 files")).toBeInTheDocument();
    const olderRow = screen.getAllByRole("row").slice(1)[1];
    expect(within(olderRow).getAllByText("-").length).toBeGreaterThan(0);
  });

  it("should edit a vector store through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-edit"));
    expect(mockOnEdit).toHaveBeenCalledWith("vs-newer");
  });

  it("should delete a vector store through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-delete"));
    expect(mockOnDelete).toHaveBeenCalledWith("vs-newer");
  });

  it("should copy the vector store ID through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("vs-newer");
  });

  describe("URL table state", () => {
    const makeStore = (vectorStoreId: string, createdAt: string): VectorStore => ({
      vector_store_id: vectorStoreId,
      custom_llm_provider: "openai",
      created_at: createdAt,
      updated_at: createdAt,
    });
    const newestB = makeStore("vs-b", "2024-03-01T00:00:00Z");
    const oldestA = makeStore("vs-a", "2024-01-01T00:00:00Z");
    const manyStores = Array.from({ length: 30 }, (_, index) =>
      makeStore(`vs-${String(index).padStart(2, "0")}`, `2024-01-01T00:00:${String(59 - index).padStart(2, "0")}Z`),
    );
    const rowIds = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getByRole("button", { name: /^vs-/ }).textContent);
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];

    it("orders rows by the sort in the URL", () => {
      renderWithProviders(<VectorStoreTable {...defaultProps} data={[newestB, oldestA]} />, {
        searchParams: "?sort_by=created_at&sort_order=asc",
      });
      expect(rowIds()).toEqual(["vs-a", "vs-b"]);
    });

    it.each(["vector_store_id", "vector_store_name", "updated_at"])(
      "honors sort_by=%s from the URL instead of falling back to created_at",
      (sortBy) => {
        const olderZulu: VectorStore = {
          ...makeStore("vs-z", "2024-01-01T00:00:00Z"),
          vector_store_name: "zulu",
          updated_at: "2024-06-01T00:00:00Z",
        };
        const newerAlpha: VectorStore = {
          ...makeStore("vs-a", "2024-03-01T00:00:00Z"),
          vector_store_name: "alpha",
          updated_at: "2024-04-01T00:00:00Z",
        };
        renderWithProviders(<VectorStoreTable {...defaultProps} data={[olderZulu, newerAlpha]} />, {
          searchParams: `?sort_by=${sortBy}&sort_order=asc`,
        });
        expect(rowIds()).toEqual(["vs-a", "vs-z"]);
      },
    );

    it("writes the sort to the URL when a header is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<VectorStoreTable {...defaultProps} data={[oldestA, newestB]} />, { onUrlUpdate });
      expect(rowIds()).toEqual(["vs-b", "vs-a"]);

      await user.click(screen.getByTestId("sort-header-vector_store_id"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_by")).toBe("vector_store_id"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_order")).toBe("asc");
      expect(rowIds()).toEqual(["vs-a", "vs-b"]);
    });

    it("opens the page named in the URL", () => {
      renderWithProviders(<VectorStoreTable {...defaultProps} data={manyStores} />, { searchParams: "?page=2" });
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
      expect(rowIds()).toEqual(["vs-25", "vs-26", "vs-27", "vs-28", "vs-29"]);
    });

    it("writes the page to the URL when paging forward", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<VectorStoreTable {...defaultProps} data={manyStores} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
      expect(rowIds()[0]).toBe("vs-25");
    });
  });
});
