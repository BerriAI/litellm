import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import FilterComponent, { FilterOption } from "./filter";

describe("FilterComponent", () => {
  const mockOnApplyFilters = vi.fn();
  const mockOnResetFilters = vi.fn();

  const defaultOptions: FilterOption[] = [
    {
      name: "teamId",
      label: "Team ID",
      options: [
        { label: "Team 1", value: "team1" },
        { label: "Team 2", value: "team2" },
      ],
    },
    {
      name: "status",
      label: "Status",
      options: [
        { label: "Active", value: "active" },
        { label: "Inactive", value: "inactive" },
      ],
    },
    {
      name: "userId",
      label: "User ID",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );
    expect(screen.getByRole("button", { name: "Filters" })).toBeInTheDocument();
  });

  it("should display custom button label", () => {
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
        buttonLabel="Custom Filters"
      />,
    );
    expect(screen.getByRole("button", { name: "Custom Filters" })).toBeInTheDocument();
  });

  it("should toggle filters visibility when filter button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    expect(screen.queryByPlaceholderText("Enter User ID...")).not.toBeInTheDocument();

    await user.click(filterButton);

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Enter User ID...")).toBeInTheDocument();
    });

    await user.click(filterButton);

    await waitFor(() => {
      expect(screen.queryByPlaceholderText("Enter User ID...")).not.toBeInTheDocument();
    });
  });

  it("should call onResetFilters when reset button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
        initialValues={{ teamId: "team1", status: "active" }}
      />,
    );

    const resetButton = screen.getByRole("button", { name: "Reset Filters" });
    await user.click(resetButton);

    await waitFor(() => {
      expect(mockOnResetFilters).toHaveBeenCalledTimes(1);
    });
  });

  it("should render filters in correct order", async () => {
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      { name: "model", label: "Model" },
      { name: "teamId", label: "Team ID" },
      { name: "status", label: "Status" },
      { name: "userId", label: "User ID" },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      const labels = screen.getAllByText(/^(Team ID|Status|User ID|Model)$/);
      expect(labels[0]).toHaveTextContent("Team ID");
      expect(labels[1]).toHaveTextContent("Status");
    });
  });

  it("should handle input filter changes", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    const userIdInput = screen.getByPlaceholderText("Enter User ID...");
    await user.type(userIdInput, "user123");

    await waitFor(() => {
      expect(mockOnApplyFilters).toHaveBeenCalledWith({ userId: "user123" });
    });
  });

  it("should display initial values in filters", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
        initialValues={{ teamId: "team1", userId: "user123" }}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      const userIdInput = screen.getByPlaceholderText("Enter User ID...") as HTMLInputElement;
      expect(userIdInput.value).toBe("user123");
    });
  });

  it("should handle select dropdown filter changes", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    const teamIdLabel = screen.getByText("Team ID");
    const teamIdSection = teamIdLabel.closest("div");
    const teamIdSelect = within(teamIdSection!).getByRole("combobox");

    await user.click(teamIdSelect);

    await waitFor(() => {
      expect(screen.getByText("Team 1")).toBeInTheDocument();
    });

    await user.click(screen.getByText("Team 1"));

    await waitFor(() => {
      expect(mockOnApplyFilters).toHaveBeenCalledWith({ teamId: "team1" });
    });
  });

  it("should handle searchable filter with search function", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([
      { label: "Result 1", value: "result1" },
      { label: "Result 2", value: "result2" },
    ]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    const modelLabel = screen.getByText("Model");
    const modelSection = modelLabel.closest("div");
    const modelSelect = within(modelSection!).getByRole("combobox");
    await user.click(modelSelect);

    await waitFor(() => {
      expect(screen.getByText("Result 1")).toBeInTheDocument();
      expect(screen.getByText("Result 2")).toBeInTheDocument();
    });
  });

  it("should debounce search input for searchable filters", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([
      { label: "Result", value: "result" },
    ]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    vi.clearAllMocks();

    const modelLabel = screen.getByText("Model");
    const modelSection = modelLabel.closest("div");
    const modelSelect = within(modelSection!).getByRole("combobox");
    await user.click(modelSelect);
    await user.type(modelSelect, "test");

    expect(mockSearchFn).not.toHaveBeenCalled();

    await waitFor(
      () => {
        expect(mockSearchFn).toHaveBeenCalledWith("test");
      },
      { timeout: 500 },
    );
  });

  it("should show loading state when searching", async () => {
    const user = userEvent.setup({ delay: null });
    let resolveSearch: (value: Array<{ label: string; value: string }>) => void;
    const mockSearchFn = vi.fn().mockImplementation(
      () =>
        new Promise<Array<{ label: string; value: string }>>((resolve) => {
          resolveSearch = resolve;
        }),
    );

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    const modelLabel = screen.getByText("Model");
    const modelSection = modelLabel.closest("div");
    const modelSelect = within(modelSection!).getByRole("combobox");
    await user.click(modelSelect);
    await user.type(modelSelect, "test");

    await waitFor(
      () => {
        expect(screen.getByText("Loading...")).toBeInTheDocument();
      },
      { timeout: 500 },
    );

    resolveSearch!([{ label: "Result", value: "result" }]);

    await waitFor(() => {
      expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    });
  });

  it("should handle search errors gracefully", async () => {
    const user = userEvent.setup({ delay: null });
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    const mockSearchFn = vi.fn().mockRejectedValue(new Error("Search failed"));

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    const modelLabel = screen.getByText("Model");
    const modelSection = modelLabel.closest("div");
    const modelSelect = within(modelSection!).getByRole("combobox");
    await user.click(modelSelect);
    await user.type(modelSelect, "test");

    await waitFor(
      () => {
        expect(consoleErrorSpy).toHaveBeenCalledWith("Error searching:", expect.any(Error));
        expect(screen.getByText("No results found")).toBeInTheDocument();
      },
      { timeout: 500 },
    );

    consoleErrorSpy.mockRestore();
  });

  it("should load initial options when dropdown opens for searchable filter", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([
      { label: "Initial Result", value: "initial" },
    ]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    vi.clearAllMocks();

    const modelLabel = screen.getByText("Model");
    const modelSection = modelLabel.closest("div");
    const modelSelect = within(modelSection!).getByRole("combobox");
    await user.click(modelSelect);

    await waitFor(() => {
      expect(screen.getByText("Initial Result")).toBeInTheDocument();
    });
  });

  it("renders caller-defined filter options that are not in PRIORITY_FILTER_ORDER (LIT-3151)", async () => {
    // Before LIT-3151, FilterComponent silently dropped any option whose name/label
    // was not in the hardcoded whitelist (e.g. "Unknown Filter" below), which made
    // the Filters panel on Tool Policies open to an empty grid because none of its
    // option names matched. The new contract: every caller-supplied option renders.
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      {
        name: "unknownFilter",
        label: "Unknown Filter",
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(screen.getByText("Unknown Filter")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter Unknown Filter...")).toBeInTheDocument();
    });
  });

  it("should call onApplyFilters with updated values when multiple filters change", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    const userIdInput = screen.getByPlaceholderText("Enter User ID...");
    await user.type(userIdInput, "user123");

    await waitFor(() => {
      expect(mockOnApplyFilters).toHaveBeenCalledWith({ userId: "user123" });
    });

    const teamIdLabel = screen.getByText("Team ID");
    const teamIdSection = teamIdLabel.closest("div");
    const teamIdSelect = within(teamIdSection!).getByRole("combobox");
    await user.click(teamIdSelect);

    await waitFor(() => {
      expect(screen.getByText("Team 1")).toBeInTheDocument();
    });

    await user.click(screen.getByText("Team 1"));

    await waitFor(() => {
      expect(mockOnApplyFilters).toHaveBeenCalledWith({
        userId: "user123",
        teamId: "team1",
      });
    });
  });

  it("should reset all filter values when reset button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(
      <FilterComponent
        options={defaultOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
        initialValues={{ teamId: "team1", userId: "user123" }}
      />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      const userIdInput = screen.getByPlaceholderText("Enter User ID...") as HTMLInputElement;
      expect(userIdInput.value).toBe("user123");
    });

    const resetButton = screen.getByRole("button", { name: "Reset Filters" });
    await user.click(resetButton);

    await waitFor(() => {
      const userIdInput = screen.getByPlaceholderText("Enter User ID...") as HTMLInputElement;
      expect(userIdInput.value).toBe("");
    });
  });
  it("renders filter options whose name is not in PRIORITY_FILTER_ORDER (LIT-3151)", async () => {
    // Bug reproducer for LIT-3151. ToolPolicies passes filter names that are not
    // on the legacy whitelist; before the fix the Filters panel opened to an
    // empty grid because every option silently rendered as null.
    const user = userEvent.setup({ delay: null });
    const toolPoliciesOptions: FilterOption[] = [
      {
        name: "Input Policy",
        label: "Input Policy",
        options: [
          { label: "Allowed", value: "allowed" },
          { label: "Blocked", value: "blocked" },
        ],
      },
      {
        name: "Output Policy",
        label: "Output Policy",
        options: [
          { label: "Allowed", value: "allowed" },
          { label: "Redacted", value: "redacted" },
        ],
      },
      {
        name: "Team Name",
        label: "Team Name",
        options: [{ label: "team-a", value: "team-a" }],
      },
      {
        name: "Key Name",
        label: "Key Name",
        options: [{ label: "key-alias-1", value: "key-alias-1" }],
      },
    ];

    renderWithProviders(
      <FilterComponent
        options={toolPoliciesOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    await waitFor(() => {
      expect(screen.getByText("Input Policy")).toBeInTheDocument();
      expect(screen.getByText("Output Policy")).toBeInTheDocument();
      expect(screen.getByText("Team Name")).toBeInTheDocument();
      expect(screen.getByText("Key Name")).toBeInTheDocument();
    });
  });

  it("preserves PRIORITY_FILTER_ORDER ordering when mixed with non-priority options", async () => {
    // Whitelisted filters still come first (so view_logs / VirtualKeysTable etc.
    // keep their existing visual ordering), with caller-defined extras rendered
    // after in input order.
    const user = userEvent.setup({ delay: null });
    const mixedOptions: FilterOption[] = [
      { name: "customExtra", label: "Custom Extra" },
      { name: "status", label: "Status" },
      { name: "anotherExtra", label: "Another Extra" },
      { name: "teamId", label: "Team ID" },
    ];

    renderWithProviders(
      <FilterComponent
        options={mixedOptions}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    await waitFor(() => {
      const labels = screen
        .getAllByText(/^(Team ID|Status|Custom Extra|Another Extra)$/)
        .map((el) => el.textContent);
      // Team ID + Status come first (priority order), then caller-defined extras.
      expect(labels).toEqual(["Team ID", "Status", "Custom Extra", "Another Extra"]);
    });
  });

  it("handles a non-empty options array with zero priority matches (LIT-3151)", async () => {
    // Direct regression for the Tool Policies scenario: every option is outside
    // the whitelist. Filters panel must still render all of them.
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      { name: "Project Tier", label: "Project Tier" },
      { name: "Region", label: "Region" },
    ];

    renderWithProviders(
      <FilterComponent
        options={options}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Enter Project Tier...")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter Region...")).toBeInTheDocument();
    });
  });
});
