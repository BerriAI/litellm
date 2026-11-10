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

  it("should render tag name as non-clickable and muted for dynamic spend tags", () => {
    render(<TagTable {...defaultProps} data={[mockDynamicSpendTag]} />);
    expect(screen.queryByRole("button", { name: "dynamic-spend-tag" })).not.toBeInTheDocument();
    expect(screen.getByText("dynamic-spend-tag")).toHaveClass("text-muted-foreground");
    expect(mockOnSelectTag).not.toHaveBeenCalled();
  });

  it("should truncate long tag names and descriptions", () => {
    const longTag: Tag = {
      ...mockTag,
      name: "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15) Firefox/152.0",
      description: "A very long description that would otherwise stretch the column far beyond what users need to see",
    };
    render(<TagTable {...defaultProps} data={[longTag]} />);
    expect(screen.getByText(longTag.name)).toHaveClass("truncate", "text-primary");
    expect(screen.getByText(longTag.description as string)).toHaveClass("truncate", "max-w-72");
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

    await user.click(editItem);
    await user.click(deleteItem);

    expect(mockOnEdit).not.toHaveBeenCalled();
    expect(mockOnDelete).not.toHaveBeenCalled();
  });

  describe("filters", () => {
    const prodTag: Tag = { ...mockTag, name: "Prod-Billing", description: "Handles Invoices" };
    const devTag: Tag = { ...mockTag, name: "dev-billing", description: "Sandbox usage" };
    const prodOnlyTag: Tag = { ...mockTag, name: "prod-search", description: "Search traffic" };
    const data = [prodTag, devTag, prodOnlyTag];

    it("should narrow rows by tag name containing the text, ignoring case", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by tag name" }), "PROD");
      expect(screen.getByText("Prod-Billing")).toBeInTheDocument();
      expect(screen.getByText("prod-search")).toBeInTheDocument();
      expect(screen.queryByText("dev-billing")).not.toBeInTheDocument();
    });

    it("should match the name anywhere in the string, not only as a prefix", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by tag name" }), "billing");
      expect(screen.getByText("Prod-Billing")).toBeInTheDocument();
      expect(screen.getByText("dev-billing")).toBeInTheDocument();
      expect(screen.queryByText("prod-search")).not.toBeInTheDocument();
    });

    it("should narrow rows by description containing the text", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by description" }), "invoice");
      expect(screen.getByText("Prod-Billing")).toBeInTheDocument();
      expect(screen.queryByText("dev-billing")).not.toBeInTheDocument();
      expect(screen.queryByText("prod-search")).not.toBeInTheDocument();
    });

    it("should require both filters to match when both are set", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by tag name" }), "billing");
      await user.type(screen.getByRole("textbox", { name: "Filter by description" }), "sandbox");
      expect(screen.getByText("dev-billing")).toBeInTheDocument();
      expect(screen.queryByText("Prod-Billing")).not.toBeInTheDocument();
      expect(screen.queryByText("prod-search")).not.toBeInTheDocument();
    });

    it("should restore every row when the filters are cleared", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      const nameFilter = screen.getByRole("textbox", { name: "Filter by tag name" });
      await user.type(nameFilter, "dev");
      expect(screen.queryByText("Prod-Billing")).not.toBeInTheDocument();
      await user.clear(nameFilter);
      expect(screen.getByText("Prod-Billing")).toBeInTheDocument();
      expect(screen.getByText("dev-billing")).toBeInTheDocument();
      expect(screen.getByText("prod-search")).toBeInTheDocument();
    });

    it("should show a no-matching message rather than the empty state when nothing matches", async () => {
      const user = userEvent.setup();
      render(<TagTable {...defaultProps} data={data} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by tag name" }), "zzz");
      expect(screen.getByText("No matching tags")).toBeInTheDocument();
      expect(screen.queryByText("No tags yet")).not.toBeInTheDocument();
    });

    it("should not fail on tags without a description", async () => {
      const user = userEvent.setup();
      const noDescription: Tag = { ...mockTag, name: "bare", description: undefined };
      render(<TagTable {...defaultProps} data={[noDescription, prodTag]} />);
      await user.type(screen.getByRole("textbox", { name: "Filter by description" }), "invoice");
      expect(screen.getByText("Prod-Billing")).toBeInTheDocument();
      expect(screen.queryByText("bare")).not.toBeInTheDocument();
    });
  });
});
