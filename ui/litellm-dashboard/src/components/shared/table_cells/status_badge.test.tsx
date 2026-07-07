import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge, type StatusTone } from "./status_badge";

const dotOf = (label: string) => screen.getByText(label).querySelector("[aria-hidden]");

describe("StatusBadge", () => {
  const dotClasses: Record<StatusTone, string> = {
    success: "bg-emerald-500",
    error: "bg-red-500",
    warning: "bg-amber-500",
    neutral: "bg-gray-400",
    info: "bg-blue-500",
  };

  (Object.entries(dotClasses) as [StatusTone, string][]).forEach(([tone, dotClass]) => {
    it(`renders a ${dotClass} dot for the ${tone} tone`, () => {
      render(<StatusBadge tone={tone} label={tone} />);
      expect(dotOf(tone)?.className).toContain(dotClass);
    });
  });

  it("renders the label text inside an outline badge", () => {
    render(<StatusBadge tone="success" label="Active" />);
    const badge = screen.getByText("Active");
    expect(badge).toBeInTheDocument();
    expect(badge.dataset.variant).toBe("outline");
  });

  it("passes dataTestId through", () => {
    render(<StatusBadge tone="error" label="Blocked" dataTestId="key-status" />);
    expect(screen.getByTestId("key-status")).toHaveTextContent("Blocked");
  });
});
