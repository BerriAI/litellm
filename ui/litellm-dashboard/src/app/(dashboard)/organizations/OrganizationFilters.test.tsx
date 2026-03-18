import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import OrganizationFilters, { FilterState } from "./OrganizationFilters";

describe("OrganizationFilters", () => {
  const defaultFilters: FilterState = {
    org_id: "",
    org_alias: "",
    sort_by: "",
    sort_order: "asc",
  };

  it("should render", () => {
    const onToggleFilters = vi.fn();
    const onChange = vi.fn();
    const onReset = vi.fn();

    render(
      <OrganizationFilters
        filters={defaultFilters}
        showFilters={false}
        onToggleFilters={onToggleFilters}
        onChange={onChange}
        onReset={onReset}
      />,
    );

    expect(screen.getByPlaceholderText("Search by Organization Name")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^filters$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reset filters/i })).toBeInTheDocument();
  });

  it("should show additional filters when showFilters is true", () => {
    const onToggleFilters = vi.fn();
    const onChange = vi.fn();
    const onReset = vi.fn();

    render(
      <OrganizationFilters
        filters={defaultFilters}
        showFilters={true}
        onToggleFilters={onToggleFilters}
        onChange={onChange}
        onReset={onReset}
      />,
    );

    expect(screen.getByPlaceholderText("Search by Organization ID")).toBeInTheDocument();
  });

  it("should call onChange when organization name input changes", async () => {
    const user = userEvent.setup();
    const onToggleFilters = vi.fn();
    const onChange = vi.fn();
    const onReset = vi.fn();

    render(
      <OrganizationFilters
        filters={defaultFilters}
        showFilters={false}
        onToggleFilters={onToggleFilters}
        onChange={onChange}
        onReset={onReset}
      />,
    );

    const input = screen.getByPlaceholderText("Search by Organization Name");
    await user.type(input, "test");

    await waitFor(
      () => {
        expect(onChange).toHaveBeenCalledWith("org_alias", expect.any(String));
      },
      { timeout: 500 },
    );
  });

  it("should call onReset when reset button is clicked", async () => {
    const user = userEvent.setup();
    const onToggleFilters = vi.fn();
    const onChange = vi.fn();
    const onReset = vi.fn();

    render(
      <OrganizationFilters
        filters={defaultFilters}
        showFilters={false}
        onToggleFilters={onToggleFilters}
        onChange={onChange}
        onReset={onReset}
      />,
    );

    const resetButton = screen.getByRole("button", { name: /reset filters/i });
    await user.click(resetButton);

    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("should show badge on filters button when filters are active", () => {
    const onToggleFilters = vi.fn();
    const onChange = vi.fn();
    const onReset = vi.fn();

    const filtersWithActive: FilterState = {
      ...defaultFilters,
      org_alias: "test org",
    };

    render(
      <OrganizationFilters
        filters={filtersWithActive}
        showFilters={false}
        onToggleFilters={onToggleFilters}
        onChange={onChange}
        onReset={onReset}
      />,
    );

    const filtersButton = screen.getByRole("button", { name: /^filters$/i });
    const badgeWrapper = filtersButton.closest(".ant-badge");
    expect(badgeWrapper).toBeInTheDocument();
  });
});
