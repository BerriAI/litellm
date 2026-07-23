import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TooltipProvider } from "@/components/ui/tooltip";

import { CopyableBadge } from "./copyable_badge";

const writeText = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  writeText.mockClear();
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
});

const renderBadge = (value: string, maxWidthClassName?: string) =>
  render(
    <TooltipProvider delay={0}>
      <CopyableBadge value={value} maxWidthClassName={maxWidthClassName} />
    </TooltipProvider>,
  );

const openTooltip = (value: string) => {
  const trigger = screen.getByText(value).parentElement!;
  fireEvent.pointerEnter(trigger, { pointerType: "mouse" });
  fireEvent.mouseEnter(trigger);
};

describe("CopyableBadge", () => {
  it("renders the value truncated in the badge", () => {
    renderBadge("app-aigateway-inference-producttech-product-default");
    const label = screen.getByText("app-aigateway-inference-producttech-product-default");
    expect(label.className).toContain("truncate");
    expect(label.className).toContain("max-w-[220px]");
  });

  it("reveals the full value and copies it from the hover tooltip", async () => {
    renderBadge("team-alias-long-name");

    openTooltip("team-alias-long-name");

    const copyButton = await screen.findByRole("button", { name: "Copy team-alias-long-name" });
    fireEvent.click(copyButton);

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("team-alias-long-name"));
  });

  it("does not expose the copy affordance until hovered", () => {
    renderBadge("my-team");
    expect(screen.queryByRole("button", { name: "Copy my-team" })).not.toBeInTheDocument();
  });

  it("honors a custom max width", () => {
    renderBadge("short", "max-w-[120px]");
    expect(screen.getByText("short").className).toContain("max-w-[120px]");
  });
});
