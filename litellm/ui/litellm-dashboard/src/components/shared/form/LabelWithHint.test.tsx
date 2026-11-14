import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { labelWithHint } from "./LabelWithHint";

describe("labelWithHint", () => {
  it("exposes named information help without submitting the surrounding form", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn((event) => event.preventDefault());
    render(<form onSubmit={onSubmit}>{labelWithHint("App client ID", "Use the public client ID.", "info")}</form>);

    const help = screen.getByRole("button", { name: "App client ID information" });
    await user.tab();
    expect(help).toHaveFocus();
    expect(await screen.findByText("Use the public client ID.")).toBeInTheDocument();

    await user.click(help);
    expect(onSubmit).not.toHaveBeenCalled();
  });
});
