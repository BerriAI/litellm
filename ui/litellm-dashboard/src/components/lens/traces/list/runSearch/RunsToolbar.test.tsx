import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { withNuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";

import { runs } from "./__fixtures__/runs";
import { RunsToolbar } from "./RunsToolbar";

const agentBox = () => screen.getByRole("combobox", { name: "Filter traces by agent" });

describe("RunsToolbar", () => {
  it("narrows the agent filter as you type and selects the match", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    render(<RunsToolbar query="" onQueryChange={vi.fn()} runs={runs} />, {
      wrapper: withNuqsTestingAdapter({ onUrlUpdate }),
    });
    expect(agentBox()).toHaveAttribute("placeholder", "All agents");
    await user.click(agentBox());
    await user.keyboard("tri");
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual(["triage"]);
    await user.click(screen.getByRole("option", { name: "triage" }));
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("agent")).toBe("triage");
  });

  it("picks the first match on Enter", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    render(<RunsToolbar query="" onQueryChange={vi.fn()} runs={runs} />, {
      wrapper: withNuqsTestingAdapter({ onUrlUpdate }),
    });
    await user.click(agentBox());
    await user.keyboard("res{Enter}");
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("agent")).toBe("researcher");
  });

  it("says when no agent matches", async () => {
    const user = userEvent.setup();
    render(<RunsToolbar query="" onQueryChange={vi.fn()} runs={runs} />, { wrapper: withNuqsTestingAdapter() });
    await user.click(agentBox());
    await user.keyboard("zzz");
    expect(screen.getByText("No matching agents")).toBeVisible();
  });

  it("clears back to all agents", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    render(<RunsToolbar query="" onQueryChange={vi.fn()} runs={runs} />, {
      wrapper: withNuqsTestingAdapter({ searchParams: "?agent=triage", onUrlUpdate }),
    });
    expect(agentBox()).toHaveValue("triage");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("agent")).toBeNull();
  });
});
