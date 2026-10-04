import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { runs } from "./__fixtures__/runs";
import { RunSearch } from "./RunSearch";

const box = () => screen.getByRole("combobox", { name: "Search runs" });
const listbox = () => screen.getByRole("listbox", { name: "Search suggestions" });

describe("RunSearch", () => {
  it("offers the run fields, then the agents seen in the loaded runs", async () => {
    const user = userEvent.setup();
    render(<RunSearch value="" onChange={vi.fn()} runs={runs} />);
    expect(screen.getByText("Search runs, or filter like agent:researcher status:error")).toBeVisible();
    await user.click(box());
    expect(
      within(listbox())
        .getAllByRole("group")
        .map((g) => g.getAttribute("aria-label")),
    ).toEqual(["Run attributes", "Content", "Identity"]);
    await user.keyboard("agent:");
    expect(
      within(listbox())
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual(["billing-agent", "cron", "researcher", "triage"]);
    await user.click(within(listbox()).getByRole("option", { name: "researcher" }));
    expect(box()).toHaveTextContent(/^agent:researcher $/, { normalizeWhitespace: false });
  });
});
