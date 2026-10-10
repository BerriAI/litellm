import { PencilAltIcon } from "@heroicons/react/outline";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type React from "react";
import { describe, expect, it, vi } from "vitest";
import BaseActionButton from "./BaseActionButton";

describe("BaseActionButton", () => {
  it("should render", () => {
    const onClick = vi.fn();
    render(<BaseActionButton icon={PencilAltIcon} onClick={onClick} dataTestId="test-button" />);
    expect(screen.getByTestId("test-button")).toBeInTheDocument();
  });

  it("should call onClick when clicked", () => {
    const onClick = vi.fn();
    render(<BaseActionButton icon={PencilAltIcon} onClick={onClick} dataTestId="test-button" />);
    const button = screen.getByTestId("test-button");

    act(() => {
      fireEvent.click(button);
    });

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("is a labelled button reachable by keyboard", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<BaseActionButton icon={PencilAltIcon} onClick={onClick} label="Edit member" />);

    await user.tab();
    expect(screen.getByRole("button", { name: "Edit member" })).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not submit an enclosing form", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event: React.FormEvent) => event.preventDefault());
    render(
      <form onSubmit={onSubmit}>
        <BaseActionButton icon={PencilAltIcon} onClick={() => undefined} label="Edit" />
      </form>,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));

    expect(onSubmit).not.toHaveBeenCalled();
  });
});
