import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { StatusBadge, type StatusTone } from "./status_badge";

describe("StatusBadge", () => {
  const toneClasses: Record<StatusTone, string[]> = {
    success: ["border-green-200", "bg-green-50", "text-green-600"],
    error: ["border-red-200", "bg-red-50", "text-red-600"],
    warning: ["border-amber-200", "bg-amber-50", "text-amber-600"],
    neutral: ["border-gray-200", "bg-gray-50", "text-gray-600"],
    info: ["border-blue-200", "bg-blue-50", "text-blue-600"],
  };

  (Object.entries(toneClasses) as [StatusTone, string[]][]).forEach(([tone, classes]) => {
    it(`renders a tinted pill (${classes.join(" ")}) for the ${tone} tone`, () => {
      render(<StatusBadge tone={tone} label={tone} />);
      const badge = screen.getByText(tone);
      classes.forEach((cls) => expect(badge.className).toContain(cls));
    });
  });

  it("renders the label text inside an outline badge with no status dot", () => {
    render(<StatusBadge tone="success" label="Active" />);
    const badge = screen.getByText("Active");
    expect(badge.dataset.variant).toBe("outline");
    expect(badge.querySelector("[aria-hidden]")).toBeNull();
  });

  it("passes dataTestId through", () => {
    render(<StatusBadge tone="error" label="Blocked" dataTestId="key-status" />);
    expect(screen.getByTestId("key-status")).toHaveTextContent("Blocked");
  });

  it("opens the tooltip on hover, which requires Badge to forward its ref to the trigger", async () => {
    const user = userEvent.setup();
    render(<StatusBadge tone="error" label="Blocked" tooltip="This key was blocked by SCIM" />);
    await user.hover(screen.getByText("Blocked"));
    expect(await screen.findByText("This key was blocked by SCIM")).toBeInTheDocument();
  });
});
