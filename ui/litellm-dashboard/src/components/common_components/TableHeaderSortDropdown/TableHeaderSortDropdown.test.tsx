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

    await user.click(screen.getByRole("button"));

    await waitFor(() => {
      expect(screen.getByRole("menuitem", { name: /ascending/i })).toBeInTheDocument();
      expect(screen.getByRole("menuitem", { name: /descending/i })).toBeInTheDocument();
      expect(screen.getByRole("menuitem", { name: /reset/i })).toBeInTheDocument();
    });
  });

  it("should call onSortChange with asc when ascending option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    await user.click(screen.getByRole("button"));
    await user.click(await screen.findByRole("menuitem", { name: /ascending/i }));

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith("asc");
  });

  it("should call onSortChange with desc when descending option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />);

    await user.click(screen.getByRole("button"));
    await user.click(await screen.findByRole("menuitem", { name: /descending/i }));

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith("desc");
  });

  it("should call onSortChange with false when reset option is clicked", async () => {
    const user = userEvent.setup();
    const onSortChange = vi.fn();
    render(<TableHeaderSortDropdown sortState="asc" onSortChange={onSortChange} />);

    await user.click(screen.getByRole("button"));
    await user.click(await screen.findByRole("menuitem", { name: /reset/i }));

    expect(onSortChange).toHaveBeenCalledTimes(1);
    expect(onSortChange).toHaveBeenCalledWith(false);
  });

  it("should visually indicate ascending sort on the trigger when sort state is asc", () => {
    const onSortChange = vi.fn();
    const { container } = render(
      <TableHeaderSortDropdown sortState="asc" onSortChange={onSortChange} />,
    );
    // The trigger is colored text-primary when sorted, not text-muted-foreground
    const button = screen.getByRole("button");
    expect(button.className).toMatch(/text-primary/);
    // And the chevron-up icon is rendered
    expect(container.querySelector("svg.lucide-chevron-up")).toBeInTheDocument();
  });

  it("should visually indicate descending sort on the trigger when sort state is desc", () => {
    const onSortChange = vi.fn();
    const { container } = render(
      <TableHeaderSortDropdown sortState="desc" onSortChange={onSortChange} />,
    );
    const button = screen.getByRole("button");
    expect(button.className).toMatch(/text-primary/);
    expect(container.querySelector("svg.lucide-chevron-down")).toBeInTheDocument();
  });

  it("should not visually indicate sort on the trigger when sort state is false", () => {
    const onSortChange = vi.fn();
    const { container } = render(
      <TableHeaderSortDropdown sortState={false} onSortChange={onSortChange} />,
    );
    const button = screen.getByRole("button");
    expect(button.className).toMatch(/text-muted-foreground/);
    expect(container.querySelector("svg.lucide-arrow-up-down")).toBeInTheDocument();
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

    await user.click(screen.getByRole("button"));

    expect(onParentClick).not.toHaveBeenCalled();
  });
});
