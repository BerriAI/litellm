import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { ToolbarSeparator } from "./ToolbarSeparator";

function renderSeparator(className?: string): HTMLElement {
  const { container } = render(<ToolbarSeparator className={className} />);
  const separator = container.querySelector('[data-slot="separator"]');
  if (!(separator instanceof HTMLElement)) {
    throw new Error("ToolbarSeparator did not render a separator element");
  }
  return separator;
}

describe("ToolbarSeparator", () => {
  it("drops the primitive's self-stretch so a fixed-height divider stays vertically centered", () => {
    const separator = renderSeparator();

    expect(separator.className).not.toMatch(/self-stretch/);
    expect(separator.className).toContain("data-vertical:self-center");
  });

  it("stays vertical and keeps its fixed height", () => {
    const separator = renderSeparator();

    expect(separator).toHaveAttribute("data-orientation", "vertical");
    expect(separator.className).toContain("h-5");
  });

  it("lets callers override spacing without resurrecting self-stretch", () => {
    const separator = renderSeparator("mx-0.5");

    expect(separator.className).toContain("mx-0.5");
    expect(separator.className).not.toMatch(/mx-1\.5/);
    expect(separator.className).not.toMatch(/self-stretch/);
  });
});
