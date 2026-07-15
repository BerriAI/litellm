import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { formatCellDate } from "@/components/shared/table_cells";
import { Tag } from "@/components/tag_management/types";

import TagTable from "./TagTable";

describe("TagTable", () => {
  const mockOnEdit = vi.fn();
  const mockOnDelete = vi.fn();
  const mockOnSelectTag = vi.fn();

  const mockTag: Tag = {
    name: "test-tag",
    description: "Test description",
    models: ["model-1", "model-2"],
    model_info: {
      "model-1": "GPT-4",
      "model-2": "Claude-3",
    },
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  };

  const mockDynamicSpendTag: Tag = {
    name: "dynamic-spend-tag",
    description:
      "This is just a spend tag that was passed dynamically in a request. It does not control any LLM models.",
    models: [],
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  };

  const defaultProps = {
    data: [],
    onEdit: mockOnEdit,
    onDelete: mockOnDelete,
    onSelectTag: mockOnSelectTag,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    render(<TagTable {...defaultProps} />);
    for (const header of ["Tag Name", "Description", "Allowed Models", "Created"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display the empty state when data is empty", () => {
    render(<TagTable {...defaultProps} />);
    expect(screen.getByText("No tags yet")).toBeInTheDocument();
  });

  it("should display tag name and description", () => {
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    expect(screen.getByText("test-tag")).toBeInTheDocument();
    expect(screen.getByText("Test description")).toBeInTheDocument();
  });

  it("should display model names from model_info", () => {
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    expect(screen.getByText("GPT-4")).toBeInTheDocument();
    expect(screen.getByText("Claude-3")).toBeInTheDocument();
  });

  it("should display All Models badge when models array is empty", () => {
    const tagWithNoModels: Tag = {
      ...mockTag,
      models: [],
    };
    render(<TagTable {...defaultProps} data={[tagWithNoModels]} />);
    expect(screen.getByText("All Models")).toBeInTheDocument();
  });

  it("should display formatted created date", () => {
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    const formattedDate = formatCellDate(new Date(mockTag.created_at), "date");
    expect(screen.getByText(formattedDate)).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    const olderTag: Tag = { ...mockTag, name: "older-tag", created_at: "2023-01-01T00:00:00Z" };
    const newerTag: Tag = { ...mockTag, name: "newer-tag", created_at: "2025-01-01T00:00:00Z" };
    render(<TagTable {...defaultProps} data={[olderTag, newerTag]} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-tag")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-tag")).toBeInTheDocument();
  });

  it("should call onSelectTag when tag name is clicked", async () => {
    const user = userEvent.setup();
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    await user.click(screen.getByRole("button", { name: "test-tag" }));
    expect(mockOnSelectTag).toHaveBeenCalledWith("test-tag");
  });

  it("should render tag name as non-clickable for dynamic spend tags", () => {
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    expect(screen.queryByRole("button", { name: "dynamic-spend-tag" })).not.toBeInTheDocument();
    expect(screen.getByText("dynamic-spend-tag")).toBeInTheDocument();
    expect(mockOnSelectTag).not.toHaveBeenCalled();
  });

  it("should edit a tag through the actions menu", async () => {
    const user = userEvent.setup();
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    await user.click(screen.getByTestId("tag-actions-test-tag"));
    await user.click(await screen.findByTestId("tag-action-edit"));
    expect(mockOnEdit).toHaveBeenCalledWith(mockTag);
  });

  it("should delete a tag through the actions menu", async () => {
    const user = userEvent.setup();
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    await user.click(screen.getByTestId("tag-actions-test-tag"));
    await user.click(await screen.findByTestId("tag-action-delete"));
    expect(mockOnDelete).toHaveBeenCalledWith("test-tag");
  });

  it("should disable edit and delete for dynamic spend tags", async () => {
    const user = userEvent.setup();
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    await user.click(screen.getByTestId("tag-actions-dynamic-spend-tag"));

    const editItem = await screen.findByTestId("tag-action-edit");
    const deleteItem = await screen.findByTestId("tag-action-delete");

    expect(editItem).toHaveAttribute("data-disabled");
    expect(deleteItem).toHaveAttribute("data-disabled");
    expect(mockOnEdit).not.toHaveBeenCalled();
    expect(mockOnDelete).not.toHaveBeenCalled();
  });
});
