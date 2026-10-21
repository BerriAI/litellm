import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
    render(<VectorStoreTable {...defaultProps} />);
    for (const header of [
      "Vector Store ID",
      "Name",
      "Description",
      "Source",
      "Files",
      "Provider",
      "Created At",
      "Updated At",
    ]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display the empty state when data is empty", () => {
    render(<VectorStoreTable {...defaultProps} data={[]} />);
    expect(screen.getByText("No vector stores")).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    render(<VectorStoreTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("vs-newer")).toBeInTheDocument();
    expect(within(rows[1]).getByText("vs-older")).toBeInTheDocument();
  });

  it("should call onView when the vector store ID is clicked", async () => {
    const user = userEvent.setup();
    render(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "vs-newer" }));
    expect(mockOnView).toHaveBeenCalledWith("vs-newer");
  });

  it("should render provider display names", () => {
    render(<VectorStoreTable {...defaultProps} />);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
  });

  it("should summarize ingested files and fall back to a dash without files", () => {
    render(<VectorStoreTable {...defaultProps} />);
    expect(screen.getByText("2 files")).toBeInTheDocument();
    const olderRow = screen.getAllByRole("row").slice(1)[1];
    expect(within(olderRow).getAllByText("-").length).toBeGreaterThan(0);
  });

  it("should edit a vector store through the actions menu", async () => {
    const user = userEvent.setup();
    render(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-edit"));
    expect(mockOnEdit).toHaveBeenCalledWith("vs-newer");
  });

  it("should delete a vector store through the actions menu", async () => {
    const user = userEvent.setup();
    render(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-delete"));
    expect(mockOnDelete).toHaveBeenCalledWith("vs-newer");
  });

  it("should label each row's source as Config or DB", () => {
    const configStore: VectorStore = { ...mockVectorStores[1], vector_store_id: "vs-config", is_config: true };
    render(<VectorStoreTable {...defaultProps} data={[mockVectorStores[0], configStore]} />);
    const rows = screen.getAllByRole("row").slice(1);
    const dbRow = rows.find((row) => within(row).queryByText("vs-newer"));
    const configRow = rows.find((row) => within(row).queryByText("vs-config"));
    expect(within(dbRow!).getByText("DB")).toBeInTheDocument();
    expect(within(dbRow!).queryByText("Config")).not.toBeInTheDocument();
    expect(within(configRow!).getByText("Config")).toBeInTheDocument();
    expect(within(configRow!).queryByText("DB")).not.toBeInTheDocument();
  });

  it("should keep edit and delete disabled for a config-defined store while copy still works", async () => {
    const user = userEvent.setup();
    const configStore: VectorStore = { ...mockVectorStores[1], vector_store_id: "vs-config", is_config: true };
    render(<VectorStoreTable {...defaultProps} data={[mockVectorStores[0], configStore]} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-config"));
    const editItem = await screen.findByTestId("vector-store-action-edit");
    const deleteItem = screen.getByTestId("vector-store-action-delete");
    expect(editItem).toHaveAttribute("aria-disabled", "true");
    expect(deleteItem).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText(/Read only: this vector store is defined in the config file/)).toBeVisible();
    await user.click(editItem);
    await user.click(deleteItem);
    expect(mockOnEdit).not.toHaveBeenCalled();
    expect(mockOnDelete).not.toHaveBeenCalled();
    await user.click(screen.getByTestId("vector-store-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("vs-config");
  });

  it("should copy the vector store ID through the actions menu", async () => {
    const user = userEvent.setup();
    render(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await user.click(await screen.findByTestId("vector-store-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("vs-newer");
  });

  it("should not show the read-only hint for a database-backed store", async () => {
    const user = userEvent.setup();
    render(<VectorStoreTable {...defaultProps} />);
    await user.click(screen.getByTestId("vector-store-actions-vs-newer"));
    await screen.findByTestId("vector-store-action-edit");
    expect(screen.queryByText(/Read only: this vector store is defined in the config file/)).not.toBeInTheDocument();
  });
});
