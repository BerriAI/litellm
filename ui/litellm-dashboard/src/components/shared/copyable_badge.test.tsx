import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CopyableBadge } from "./copyable_badge";

const writeText = vi.fn().mockResolvedValue(undefined);

beforeEach(() => {
  writeText.mockClear();
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
});

describe("CopyableBadge", () => {
  it("renders the value and truncates it", () => {
    render(<CopyableBadge value="app-aigateway-inference-producttech-product-default" />);
    const label = screen.getByText("app-aigateway-inference-producttech-product-default");
    expect(label.className).toContain("truncate");
    expect(label.className).toContain("max-w-[220px]");
  });

  it("copies the full value when the copy button is clicked", async () => {
    render(<CopyableBadge value="team-alias-long-name" />);

    fireEvent.click(screen.getByRole("button", { name: "Copy team-alias-long-name" }));

    await waitFor(() => expect(writeText).toHaveBeenCalledWith("team-alias-long-name"));
  });

  it("exposes a copy affordance via accessible label", () => {
    render(<CopyableBadge value="my-team" />);
    expect(screen.getByRole("button", { name: "Copy my-team" })).toBeInTheDocument();
  });

  it("honors a custom max width", () => {
    render(<CopyableBadge value="short" maxWidthClassName="max-w-[120px]" />);
    expect(screen.getByText("short").className).toContain("max-w-[120px]");
  });
});
