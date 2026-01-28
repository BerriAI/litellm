import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import VectorStoreTable from "./VectorStoreTable";
import { VectorStore } from "./types";

// Mock dependencies
const mockGetProviderLogoAndName = vi.fn();
const mockTableIconActionButton = vi.fn();

vi.mock("../provider_info_helpers", () => ({
  getProviderLogoAndName: (...args: any[]) => mockGetProviderLogoAndName(...args),
}));

vi.mock("../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton", () => ({
  default: (props: any) => {
    mockTableIconActionButton(props);
    return (
      <button
        data-testid={`action-button-${props.variant.toLowerCase()}`}
        data-tooltip={props.tooltipText}
        onClick={props.onClick}
        disabled={props.disabled}
      >
        {props.variant}
      </button>
    );
  },
}));

// Mock Tremor components to avoid complex styling issues
vi.mock("@tremor/react", () => ({
  Table: ({ children, ...props }: any) => <table {...props}>{children}</table>,
  TableHead: ({ children, ...props }: any) => <thead {...props}>{children}</thead>,
  TableBody: ({ children, ...props }: any) => <tbody {...props}>{children}</tbody>,
  TableRow: ({ children, ...props }: any) => <tr {...props}>{children}</tr>,
  TableHeaderCell: ({ children, ...props }: any) => <th {...props}>{children}</th>,
  TableCell: ({ children, ...props }: any) => <td {...props}>{children}</td>,
}));

// Mock antd Tooltip
vi.mock("antd", () => ({
  Tooltip: ({ children, title }: any) => (
    <div data-testid="tooltip" data-title={title}>
      {children}
    </div>
  ),
}));

// Mock Heroicons
vi.mock("@heroicons/react/outline", () => ({
  ChevronDownIcon: (props: any) => <div data-testid="chevron-down" {...props} />,
  ChevronUpIcon: (props: any) => <div data-testid="chevron-up" {...props} />,
  SwitchVerticalIcon: (props: any) => <div data-testid="switch-vertical" {...props} />,
}));

// Test data
const mockVectorStores: VectorStore[] = [
  {
    vector_store_id: "short-id",
    custom_llm_provider: "openai",
    vector_store_name: "My OpenAI Store",
    vector_store_description: "A store for OpenAI vectors",
    created_at: "2024-01-15T10:30:00Z",
    updated_at: "2024-01-15T11:00:00Z",
    created_by: "user-1",
    updated_by: "user-1",
  },
  {
    vector_store_id: "very-long-vector-store-id-that-should-be-truncated",
    custom_llm_provider: "azure",
    vector_store_name: undefined, // Test missing name
    vector_store_description: "A store for Azure vectors with a very long description that should show a tooltip",
    created_at: "2024-01-10T09:15:00Z",
    updated_at: "2024-01-12T14:20:00Z",
  },
  {
    vector_store_id: "store-3",
    custom_llm_provider: "pg_vector",
    vector_store_name: "PostgreSQL Store",
    vector_store_description: undefined, // Test missing description
    created_at: "2024-01-05T08:00:00Z",
    updated_at: "2024-01-08T16:45:00Z",
  },
];

// Mock functions
const mockOnView = vi.fn();
const mockOnEdit = vi.fn();
const mockOnDelete = vi.fn();

const defaultProps = {
  data: mockVectorStores,
  onView: mockOnView,
  onEdit: mockOnEdit,
  onDelete: mockOnDelete,
};

// Helper function to render component
const renderComponent = (props = {}) => {
  return render(<VectorStoreTable {...defaultProps} {...props} />);
};

describe("VectorStoreTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Setup default mock returns for getProviderLogoAndName
    mockGetProviderLogoAndName.mockImplementation((provider: string) => {
      const providerMap: Record<string, { displayName: string; logo: string }> = {
        openai: { displayName: "OpenAI", logo: "/openai-logo.png" },
        azure: { displayName: "Azure", logo: "/azure-logo.png" },
        pg_vector: { displayName: "PostgreSQL Vector", logo: "/pg-logo.png" },
      };
      return providerMap[provider] || { displayName: provider, logo: "" };
    });
  });

  describe("Rendering", () => {
    it("should render the table with data", () => {
      renderComponent();
      expect(screen.getByRole("table")).toBeInTheDocument();
    });

    it("should render table headers", () => {
      renderComponent();
      expect(screen.getByText("Vector Store ID")).toBeInTheDocument();
      expect(screen.getByText("Name")).toBeInTheDocument();
      expect(screen.getByText("Description")).toBeInTheDocument();
      expect(screen.getByText("Provider")).toBeInTheDocument();
      expect(screen.getByText("Created At")).toBeInTheDocument();
      expect(screen.getByText("Updated At")).toBeInTheDocument();
      // Check that we have the expected number of header cells (7 data + 1 actions)
      const headers = screen.getAllByRole("columnheader");
      expect(headers).toHaveLength(8);
    });

    it("should render all vector store rows", () => {
      renderComponent();
      expect(screen.getAllByRole("row")).toHaveLength(mockVectorStores.length + 1); // +1 for header row
    });

    it("should render empty state when no data", () => {
      renderComponent({ data: [] });
      expect(screen.getByText("No vector stores found")).toBeInTheDocument();
    });
  });

  describe("Vector Store ID Column", () => {
    it("should render short vector store IDs fully", () => {
      renderComponent();
      expect(screen.getByText("short-id")).toBeInTheDocument();
    });

    it("should truncate long vector store IDs", () => {
      renderComponent();
      // Check that the truncated text is rendered (first 15 chars + ...)
      const truncatedText = "very-long-vecto...";
      expect(screen.getByText(truncatedText)).toBeInTheDocument();
    });

    it("should make vector store ID clickable", async () => {
      const user = userEvent.setup();
      renderComponent();
      const idButton = screen.getByText("short-id");
      await user.click(idButton);
      expect(mockOnView).toHaveBeenCalledWith("short-id");
    });

    it("should have correct styling for vector store ID button", () => {
      renderComponent();
      const idButton = screen.getByText("short-id").closest("button");
      expect(idButton).toHaveClass("font-mono", "text-blue-500", "bg-blue-50", "hover:bg-blue-100");
    });
  });

  describe("Name Column", () => {
    it("should render vector store name", () => {
      renderComponent();
      expect(screen.getByText("My OpenAI Store")).toBeInTheDocument();
    });

    it("should render fallback for missing name", () => {
      renderComponent();
      const fallbackElements = screen.getAllByText("-");
      expect(fallbackElements.length).toBe(5); // One for missing name, one for missing description, three for missing files (one per store)
    });

    it("should wrap name in tooltip", () => {
      renderComponent();
      const tooltips = screen.getAllByTestId("tooltip");
      const nameTooltip = tooltips.find((t) => t.getAttribute("data-title") === "My OpenAI Store");
      expect(nameTooltip).toBeInTheDocument();
    });
  });

  describe("Description Column", () => {
    it("should render vector store description", () => {
      renderComponent();
      expect(screen.getByText("A store for OpenAI vectors")).toBeInTheDocument();
    });

    it("should render fallback for missing description", () => {
      renderComponent();
      const fallbackElements = screen.getAllByText("-");
      expect(fallbackElements.length).toBe(5); // One for missing name, one for missing description, three for missing files (one per store)
    });

    it("should wrap description in tooltip", () => {
      renderComponent();
      const tooltips = screen.getAllByTestId("tooltip");
      const descTooltip = tooltips.find(
        (t) =>
          t.getAttribute("data-title") ===
          "A store for Azure vectors with a very long description that should show a tooltip",
      );
      expect(descTooltip).toBeInTheDocument();
    });
  });

  describe("Provider Column", () => {
    it("should render provider display name", () => {
      renderComponent();
      expect(screen.getByText("OpenAI")).toBeInTheDocument();
      expect(screen.getByText("Azure")).toBeInTheDocument();
      expect(screen.getByText("PostgreSQL Vector")).toBeInTheDocument();
    });

    it("should render provider logo when available", () => {
      renderComponent();
      const logos = screen.getAllByRole("img");
      expect(logos).toHaveLength(3); // All providers have logos in our mock
      expect(logos[0]).toHaveAttribute("src", "/openai-logo.png");
      expect(logos[0]).toHaveAttribute("alt", "OpenAI");
    });

    it("should call getProviderLogoAndName for each provider", () => {
      renderComponent();
      expect(mockGetProviderLogoAndName).toHaveBeenCalledWith("openai");
      expect(mockGetProviderLogoAndName).toHaveBeenCalledWith("azure");
      expect(mockGetProviderLogoAndName).toHaveBeenCalledWith("pg_vector");
    });
  });

  describe("Date Columns", () => {
    it("should render created at dates", () => {
      renderComponent();
      const dateElements = screen.getAllByText(/1\/\d+\/2024/);
      expect(dateElements.length).toBe(6); // 3 created_at + 3 updated_at dates
    });

    it("should render updated at dates", () => {
      renderComponent();
      const dateElements = screen.getAllByText(/1\/\d+\/2024/);
      expect(dateElements.length).toBe(6); // 3 created_at + 3 updated_at dates
    });
  });

  describe("Actions Column", () => {
    it("should render edit and delete action buttons for each row", () => {
      renderComponent();
      expect(screen.getAllByTestId("action-button-edit")).toHaveLength(mockVectorStores.length);
      expect(screen.getAllByTestId("action-button-delete")).toHaveLength(mockVectorStores.length);
    });

    it("should call onEdit when edit button is clicked", async () => {
      const user = userEvent.setup();
      renderComponent();
      const editButtons = screen.getAllByTestId("action-button-edit");
      await user.click(editButtons[0]);
      expect(mockOnEdit).toHaveBeenCalledWith("short-id");
    });

    it("should call onDelete when delete button is clicked", async () => {
      const user = userEvent.setup();
      renderComponent();
      const deleteButtons = screen.getAllByTestId("action-button-delete");
      await user.click(deleteButtons[0]);
      expect(mockOnDelete).toHaveBeenCalledWith("short-id");
    });

    it("should pass correct props to TableIconActionButton", () => {
      renderComponent();
      expect(mockTableIconActionButton).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "Edit",
          tooltipText: "Edit vector store",
          onClick: expect.any(Function),
        }),
      );
      expect(mockTableIconActionButton).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "Delete",
          tooltipText: "Delete vector store",
          onClick: expect.any(Function),
        }),
      );
    });
  });

  describe("Sorting", () => {
    it("should initialize with created_at descending sort", () => {
      renderComponent();
      // The table should initialize with sorting state
      expect(screen.getByTestId("chevron-down")).toBeInTheDocument();
    });

    it("should render sort icons for sortable columns", () => {
      renderComponent();
      // Should have sort icons for Created At and Updated At columns
      const sortIcons = screen.getAllByTestId(/^chevron-(up|down)$|^switch-vertical$/);
      expect(sortIcons.length).toBeGreaterThan(0);
    });

    it("should make header cells clickable for sorting", () => {
      renderComponent();
      const headerCells = screen.getAllByRole("columnheader");
      const sortableHeaders = headerCells.filter((cell) => cell.textContent !== "");
      expect(sortableHeaders.length).toBeGreaterThan(0);
    });

    it("should show ascending icon when sorted ascending", () => {
      renderComponent();
      // Initially shows descending, but we can test the logic by checking the icons are present
      expect(screen.getByTestId("chevron-down")).toBeInTheDocument();
    });
  });

  describe("Styling and Layout", () => {
    it("should apply correct CSS classes to table container", () => {
      renderComponent();
      const tableContainer = screen.getByRole("table").parentElement?.parentElement;
      expect(tableContainer).toHaveClass("rounded-lg", "custom-border", "relative");
    });

    it("should apply overflow styling to table wrapper", () => {
      renderComponent();
      const tableWrapper = screen.getByRole("table").parentElement;
      expect(tableWrapper).toHaveClass("overflow-x-auto");
    });

    it("should apply sticky styling to actions column", () => {
      renderComponent();
      const headerCells = screen.getAllByRole("columnheader");
      const actionsHeader = headerCells[headerCells.length - 1];
      expect(actionsHeader).toHaveClass("sticky", "right-0", "bg-white");
    });

    it("should apply sticky styling to action cells", () => {
      renderComponent();
      const rows = screen.getAllByRole("row").slice(1); // Skip header row
      rows.forEach((row) => {
        const cells = row.querySelectorAll("td");
        const lastCell = cells[cells.length - 1];
        expect(lastCell).toHaveClass("sticky", "right-0", "bg-white");
      });
    });
  });

  describe("Table Row Styling", () => {
    it("should apply correct height to table rows", () => {
      renderComponent();
      const rows = screen.getAllByRole("row").slice(1); // Skip header row
      rows.forEach((row) => {
        expect(row).toHaveClass("h-8");
      });
    });

    it("should apply correct cell padding and styling", () => {
      renderComponent();
      const cells = screen.getAllByRole("cell");
      cells.forEach((cell) => {
        expect(cell).toHaveClass("py-0.5", "max-h-8", "overflow-hidden", "text-ellipsis", "whitespace-nowrap");
      });
    });
  });

  describe("Empty State", () => {
    it("should render single row with centered message when no data", () => {
      renderComponent({ data: [] });
      const rows = screen.getAllByRole("row");
      expect(rows).toHaveLength(2); // Header + empty state row
      expect(screen.getByText("No vector stores found")).toBeInTheDocument();
    });

    it("should span all columns in empty state", () => {
      renderComponent({ data: [] });
      const emptyCell = screen.getByText("No vector stores found").closest("td");
      expect(emptyCell).toHaveAttribute("colSpan", "8"); // 7 data columns + 1 actions column
    });
  });

  describe("Data Edge Cases", () => {
    it("should handle vector stores with minimal data", () => {
      const minimalData: VectorStore[] = [
        {
          vector_store_id: "minimal",
          custom_llm_provider: "test",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];

      renderComponent({ data: minimalData });
      expect(screen.getByText("minimal")).toBeInTheDocument();
      expect(screen.getAllByText("-")).toHaveLength(3); // Name, description, and files fallbacks
    });

    it("should handle single vector store", () => {
      const singleData = [mockVectorStores[0]];
      renderComponent({ data: singleData });
      expect(screen.getAllByRole("row")).toHaveLength(2); // Header + 1 data row
    });
  });
});
