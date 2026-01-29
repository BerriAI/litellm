import { screen, waitFor } from "@testing-library/react";
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

  it("should call onResetFilters when reset button is clicked", async () => {
    const user = userEvent.setup();
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
    const user = userEvent.setup();
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
    const user = userEvent.setup();
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
});
