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

  it("mirrors the filters as trace SQL bounded to the shown range, ready to copy as curl", async () => {
    const user = userEvent.setup();
    const range = { startMs: 1_700_000_000_000, endMs: 1_700_003_600_000 };
    render(<RunSearch value="" onChange={vi.fn()} runs={runs} range={range} />);
    await user.click(box());
    const sql = () => screen.getByLabelText("SQL equivalent").textContent;
    expect(sql()).toBe(
      "min(StartTs) >= fromUnixTimestamp64Milli(1700000000000) AND min(StartTs) < fromUnixTimestamp64Milli(1700003600000)",
    );
    await user.keyboard("agent:res");
    expect(sql()).toBe("arrayExists(x -> x ILIKE 'res', agents)");
    await user.click(screen.getByRole("button", { name: "Copy as curl" }));
    const command = await navigator.clipboard.readText();
    expect(command).toContain('/v1/traces/query"');
    expect(command).toContain("fromUnixTimestamp64Milli(1700000000000)");
    expect(command).toContain("arrayExists(x -> x ILIKE 'res', agents)");
    expect(screen.getByRole("button", { name: "Copied" })).toBeVisible();
    expect(box()).toHaveAttribute("aria-expanded", "true");
  });
});
