import { PencilAltIcon } from "@heroicons/react/outline";
import { act, fireEvent, render, screen } from "@testing-library/react";
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
});
