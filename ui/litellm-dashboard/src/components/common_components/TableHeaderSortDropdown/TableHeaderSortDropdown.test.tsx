import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TableHeaderSortDropdown } from "./TableHeaderSortDropdown";

describe("TableHeaderSortDropdown", () => {
  it("should render", () => {
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("should open dropdown menu when button is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText("Ascending")).toBeInTheDocument();
      expect(screen.getByText("Descending")).toBeInTheDocument();
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });
  });

  it("should call onSortChange with asc when ascending option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText("Ascending")).toBeInTheDocument();
    });

    const ascendingOption = screen.getByText("Ascending");
    await user.click(ascendingOption);

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith("asc");
  });

  it("should call onSortChange with desc when descending option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText("Descending")).toBeInTheDocument();
    });

    const descendingOption = screen.getByText("Descending");
    await user.click(descendingOption);

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith("desc");
  });

  it("should call onSortChange with false when reset option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState="asc" onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText("Reset")).toBeInTheDocument();
    });

    const resetOption = screen.getByText("Reset");
    await user.click(resetOption);

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith(false);
  });

  it("should highlight ascending option when sort state is asc", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState="asc" onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      const ascendingOption = screen.getByText("Ascending");
      const menuItem = ascendingOption.closest(".ant-dropdown-menu-item");
      expect(menuItem).toHaveClass("ant-dropdown-menu-item-selected");
    });
  });

  it("should highlight descending option when sort state is desc", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState="desc" onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      const descendingOption = screen.getByText("Descending");
      const menuItem = descendingOption.closest(".ant-dropdown-menu-item");
      expect(menuItem).toHaveClass("ant-dropdown-menu-item-selected");
    });
  });

  it("should not highlight any option when sort state is false", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    const button = screen.getByRole("button");
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText("Ascending")).toBeInTheDocument();
    });

    const ascendingOption = screen.getByText("Ascending");
    const menuItem = ascendingOption.closest(".ant-dropdown-menu-item");
    expect(menuItem).not.toHaveClass("ant-dropdown-menu-item-selected");
  });

  it("should stop event propagation when button is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    const onParentClick = vi.fn();

    render(
      <div onClick={onParentClick}>
        <TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />
      </div>,
    );

    const button = screen.getByRole("button");
    await user.click(button);

    expect(onParentClick).not.toHaveBeenCalled();
  });
});
