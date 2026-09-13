import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Badge } from "./badge";

describe("Badge", () => {
  it("renders a span carrying the badge slot and variant classes by default", () => {
    render(<Badge variant="outline">v1.2.3</Badge>);
    const badge = screen.getByText("v1.2.3");
    expect(badge.tagName).toBe("SPAN");
    expect(badge).toHaveAttribute("data-slot", "badge");
    expect(badge).toHaveAttribute("data-variant", "outline");
    expect(badge).toHaveClass("border-border");
  });

  it("renders as an anchor via the render prop while keeping badge styling", () => {
    render(
      <Badge variant="outline" render={<a href="https://docs.litellm.ai/release_notes" />}>
        v1.2.3
      </Badge>,
    );
    const link = screen.getByRole("link", { name: "v1.2.3" });
    expect(link.tagName).toBe("A");
    expect(link).toHaveAttribute("href", "https://docs.litellm.ai/release_notes");
    expect(link).toHaveAttribute("data-slot", "badge");
    expect(link).toHaveClass("border-border");
  });

  it("lets className win over variant classes through twMerge", () => {
    render(<Badge className="text-muted-foreground">x</Badge>);
    expect(screen.getByText("x")).toHaveClass("text-muted-foreground");
  });
});
