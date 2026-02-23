import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import TagTable from "./TagTable";
import { Tag } from "./types";

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

  it("should render", () => {
    render(<TagTable {...defaultProps} />);
    expect(screen.getByText("Tag Name")).toBeInTheDocument();
    expect(screen.getByText("Description")).toBeInTheDocument();
    expect(screen.getByText("Allowed Models")).toBeInTheDocument();
    expect(screen.getByText("Created")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should display no tags found message when data is empty", () => {
    render(<TagTable {...defaultProps} />);
    expect(screen.getByText("No tags found")).toBeInTheDocument();
  });

  it("should display tag name", () => {
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    expect(screen.getByText("test-tag")).toBeInTheDocument();
  });

  it("should display tag description", () => {
    render(<TagTable {...defaultProps} data={[mockTag]} />);
    expect(screen.getByText("Test description")).toBeInTheDocument();
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
    const formattedDate = new Date(mockTag.created_at).toLocaleDateString();
    expect(screen.getByText(formattedDate)).toBeInTheDocument();
  });

  it("should disable tag name button for dynamic spend tags", () => {
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    const tagButton = screen.getByRole("button", { name: "dynamic-spend-tag" });
    expect(tagButton).toBeDisabled();
  });

  it("should disable edit icon for dynamic spend tags", () => {
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    const editIcon = screen.getByLabelText("Edit tag (disabled)");
    expect(editIcon).toBeInTheDocument();
    expect(editIcon).toHaveClass("cursor-not-allowed");
  });

  it("should disable delete icon for dynamic spend tags", () => {
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    const deleteIcon = screen.getByLabelText("Delete tag (disabled)");
    expect(deleteIcon).toBeInTheDocument();
    expect(deleteIcon).toHaveClass("cursor-not-allowed");
  });
});
