import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DataTableToolbar } from "./DataTableToolbar";

describe("DataTableToolbar", () => {
  it("renders slotted action children", () => {
    render(
      <DataTableToolbar>
        <button data-testid="toolbar-action">Action</button>
      </DataTableToolbar>,
    );
    expect(screen.getByTestId("toolbar-action")).toBeInTheDocument();
  });

  it("shows the reset button only when there are active filters", async () => {
    const user = userEvent.setup();
    const onResetFilters = vi.fn();
    const { rerender } = render(<DataTableToolbar onResetFilters={onResetFilters} hasActiveFilters={false} />);
    expect(screen.queryByText("Reset Filters")).toBeNull();

    rerender(<DataTableToolbar onResetFilters={onResetFilters} hasActiveFilters />);
    await user.click(screen.getByText("Reset Filters"));
    expect(onResetFilters).toHaveBeenCalledTimes(1);
  });

  it("wires the filters toggle button", async () => {
    const user = userEvent.setup();
    const onToggleFilters = vi.fn();
    render(<DataTableToolbar onToggleFilters={onToggleFilters} />);
    await user.click(screen.getByText("Filters"));
    expect(onToggleFilters).toHaveBeenCalledTimes(1);
  });
});
