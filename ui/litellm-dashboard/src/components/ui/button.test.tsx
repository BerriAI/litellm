import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { Button } from "./button";

describe("Button", () => {
  it("renders the default variant with cva-applied classes", () => {
    render(<Button>Save</Button>);
    const button = screen.getByRole("button", { name: "Save" });
    expect(button).toHaveClass("bg-primary");
    expect(button).toHaveAttribute("data-slot", "button");
  });

  it("applies variant and size props", () => {
    render(
      <Button variant="destructive" size="sm">
        Delete
      </Button>,
    );
    const button = screen.getByRole("button", { name: "Delete" });
    expect(button).toHaveClass("bg-destructive/10");
    expect(button).toHaveClass("h-8");
  });

  it("resolves conflicting classes through twMerge so className wins", () => {
    render(<Button className="bg-red-500">Override</Button>);
    const button = screen.getByRole("button", { name: "Override" });
    expect(button).toHaveClass("bg-red-500");
    expect(button).not.toHaveClass("bg-primary");
  });

  it("renders the element passed via the render prop with button semantics", () => {
    render(<Button nativeButton={false} render={<a href="https://docs.litellm.ai">Docs</a>} />);
    const button = screen.getByRole("button", { name: "Docs" });
    expect(button.tagName).toBe("A");
    expect(button).toHaveAttribute("href", "https://docs.litellm.ai");
    expect(button).toHaveClass("bg-primary");
  });

  it("forwards refs to the underlying button element", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Save</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
  });
});
