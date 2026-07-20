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

  it("renders filters in the caller-supplied order", async () => {
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      { name: "model", label: "Model" },
      { name: "teamId", label: "Team ID" },
      { name: "status", label: "Status" },
      { name: "userId", label: "User ID" },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      const labels = screen.getAllByText(/^(Team ID|Status|User ID|Model)$/);
      expect(labels.map((l) => l.textContent)).toEqual(["Model", "Team ID", "Status", "User ID"]);
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
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
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
    const mockSearchFn = vi.fn().mockResolvedValue([{ label: "Result", value: "result" }]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
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
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
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

  it("shows a loading state (not an empty list) while a searchable filter's data is still loading", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        loading: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    const modelLabel = screen.getByText("Model");
    const modelSelect = within(modelLabel.closest("div")!).getByRole("combobox");
    await user.click(modelSelect);

    await waitFor(() => {
      expect(screen.getByText("Loading...")).toBeInTheDocument();
    });
    expect(screen.queryByText("No results found")).not.toBeInTheDocument();
    // It must not cache an empty initial-options list while the source is still loading.
    expect(mockSearchFn).not.toHaveBeenCalled();
  });

  it("loads initial options once a searchable filter's data finishes loading", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([{ label: "Team A", value: "team-a" }]);
    const baseOption: FilterOption = { name: "model", label: "Model", isSearchable: true, searchFn: mockSearchFn };

    const { rerender } = renderWithProviders(
      <FilterComponent
        options={[{ ...baseOption, loading: true }]}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));
    expect(mockSearchFn).not.toHaveBeenCalled();

    rerender(
      <FilterComponent
        options={[{ ...baseOption, loading: false }]}
        onApplyFilters={mockOnApplyFilters}
        onResetFilters={mockOnResetFilters}
      />,
    );

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
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
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
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
    const mockSearchFn = vi.fn().mockResolvedValue([{ label: "Initial Result", value: "initial" }]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
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

  it("renders caller-supplied options that match no predefined filter name (LIT-3151)", async () => {
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      {
        name: "unknownFilter",
        label: "Unknown Filter",
      },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
    );

    const filterButton = screen.getByRole("button", { name: "Filters" });
    await user.click(filterButton);

    await waitFor(() => {
      expect(screen.getByText("Unknown Filter")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter Unknown Filter...")).toBeInTheDocument();
    });
  });

  it("renders every Tool Policies filter when none match a predefined name (LIT-3151)", async () => {
    const user = userEvent.setup({ delay: null });
    const options: FilterOption[] = [
      { name: "Input Policy", label: "Input Policy", options: [{ label: "Trusted", value: "trusted" }] },
      { name: "Output Policy", label: "Output Policy", options: [{ label: "Blocked", value: "blocked" }] },
      { name: "Team Name", label: "Team Name", options: [] },
      { name: "Key Name", label: "Key Name", options: [] },
    ];

    renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    await waitFor(() => {
      const labels = screen.getAllByText(/^(Input Policy|Output Policy|Team Name|Key Name)$/);
      expect(labels.map((l) => l.textContent)).toEqual(["Input Policy", "Output Policy", "Team Name", "Key Name"]);
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

  it("cancels a pending debounced search when the component unmounts mid-type", async () => {
    const user = userEvent.setup({ delay: null });
    const mockSearchFn = vi.fn().mockResolvedValue([{ label: "Result", value: "result" }]);

    const options: FilterOption[] = [
      {
        name: "model",
        label: "Model",
        isSearchable: true,
        searchFn: mockSearchFn,
      },
    ];

    const { unmount } = renderWithProviders(
      <FilterComponent options={options} onApplyFilters={mockOnApplyFilters} onResetFilters={mockOnResetFilters} />,
    );

    await user.click(screen.getByRole("button", { name: "Filters" }));

    await waitFor(() => {
      expect(mockSearchFn).toHaveBeenCalledWith("");
    });

    vi.clearAllMocks();

    const modelLabel = screen.getByText("Model");
    const modelSelect = within(modelLabel.closest("div")!).getByRole("combobox");
    await user.click(modelSelect);
    await user.type(modelSelect, "test");

    expect(mockSearchFn).not.toHaveBeenCalled();

    unmount();

    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(mockSearchFn).not.toHaveBeenCalled();
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
});
