import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { StatusBadge, type StatusTone } from "./status_badge";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

describe("StatusBadge", () => {
  const toneClasses: Record<StatusTone, string[]> = {
    success: ["border-success/20", "bg-success/10", "text-success"],
    error: ["border-destructive/20", "bg-destructive/10", "text-destructive"],
    warning: ["border-warning/20", "bg-warning/10", "text-warning"],
    neutral: ["border-border", "bg-muted", "text-muted-foreground"],
    info: ["border-info/20", "bg-info/10", "text-info"],
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

  it("renders a tinted anchor that navigates client-side when href is given", async () => {
    const user = userEvent.setup();
    render(<StatusBadge tone="info" label="gpt-4.1" href="/models-and-endpoints?model_group=gpt-4.1" />);
    const link = screen.getByRole("link", { name: "gpt-4.1" });
    expect(link).toHaveAttribute("href", "/models-and-endpoints?model_group=gpt-4.1");
    expect(link).toHaveClass("text-info");
    await user.click(link);
    expect(push).toHaveBeenCalledWith("/models-and-endpoints?model_group=gpt-4.1");
  });

  it("renders no anchor without an href", () => {
    render(<StatusBadge tone="info" label="gpt-4.1" />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
