import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { runs } from "./__fixtures__/runs";
import { RunSearch } from "./RunSearch";

function Harness({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <RunSearch value={value} onChange={setValue} runs={runs} />
      <button type="button" onClick={() => setValue("status:ok")}>
        Load saved query
      </button>
    </>
  );
}

/** Delivers each emitted value only when asked, like a throttled URL. */
function LaggingHarness() {
  const [value, setValue] = useState("");
  const [queue, setQueue] = useState<string[]>([]);
  return (
    <>
      <RunSearch value={value} onChange={(next) => setQueue((q) => [...q, next])} runs={runs} />
      <output aria-label="undelivered">{queue.join("|")}</output>
      <button
        type="button"
        onClick={() => {
          setValue(queue[0]);
          setQueue(queue.slice(1));
        }}
      >
        Deliver next
      </button>
    </>
  );
}

const box = () => screen.getByRole("combobox", { name: "Search runs" });
const undelivered = () => screen.getByRole("status", { name: "undelivered" });
const listbox = () => screen.getByRole("listbox", { name: "Search suggestions" });

describe("RunSearch", () => {
  it("builds a filter from the keyboard: field, then value", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    expect(box()).toHaveAttribute("aria-expanded", "true");
    await user.keyboard("-sta");
    expect(
      within(listbox())
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual(["status"]);
    await user.keyboard("{Enter}");
    expect(box()).toHaveTextContent(/^-status:$/, { normalizeWhitespace: false });
    expect(within(listbox()).getByRole("group", { name: "status" })).toBeVisible();
    expect(within(listbox()).getByText("does not contain")).toBeVisible();

    await user.keyboard("{ArrowDown}");
    const ok = within(listbox()).getByRole("option", { name: "ok" });
    expect(ok).toHaveAttribute("aria-selected", "true");
    expect(box()).toHaveAttribute("aria-activedescendant", ok.id);
    await user.keyboard("{Enter}");
    expect(box()).toHaveTextContent(/^-status:ok $/, { normalizeWhitespace: false });
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(box()).toHaveAttribute("aria-expanded", "false");
  });

  it("wraps arrow navigation and picks with Tab", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    await user.keyboard("{ArrowUp}");
    expect(within(listbox()).getByRole("option", { name: "trace_id" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowDown}{ArrowDown}{Tab}");
    expect(box()).toHaveTextContent(/^agent:$/, { normalizeWhitespace: false });
    await user.keyboard("{ArrowDown}{ArrowDown}e");
    expect(within(listbox()).getByRole("option", { name: "billing-agent" })).toHaveAttribute("aria-selected", "true");
  });

  it("picks a suggestion with the mouse without losing focus", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    await user.keyboard("agent:");
    await user.click(within(listbox()).getByRole("option", { name: "researcher" }));
    expect(box()).toHaveTextContent(/^agent:researcher $/, { normalizeWhitespace: false });
    expect(box()).toHaveFocus();
    await user.keyboard("refund");
    expect(box()).toHaveTextContent(/^agent:researcher refund$/, { normalizeWhitespace: false });
  });

  it("stays one line: Enter with no menu adds nothing, and Escape closes the menu until ArrowDown", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    await user.keyboard("refund{Enter}");
    expect(box()).toHaveTextContent(/^refund$/, { normalizeWhitespace: false });
    await user.keyboard(" {Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    await user.keyboard("{ArrowDown}");
    expect(listbox()).toBeVisible();
  });

  it("emits the query once typing pauses, not per keystroke", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<RunSearch value="" onChange={onChange} runs={runs} />);
    await user.click(box());
    await user.keyboard("refund");
    await waitFor(() => expect(onChange).toHaveBeenCalledWith("refund"));
    expect(onChange).toHaveBeenCalledOnce();
  });

  it("keeps typing intact while the value it emitted arrives late", async () => {
    const user = userEvent.setup();
    render(<LaggingHarness />);
    await user.click(box());
    await user.keyboard("ab");
    await waitFor(() => expect(undelivered()).toHaveTextContent(/^ab$/, { normalizeWhitespace: false }));
    await user.keyboard("c");
    await user.click(screen.getByRole("button", { name: "Deliver next" }));
    await user.click(box());
    await user.keyboard("d");
    expect(box()).toHaveTextContent(/^abcd$/, { normalizeWhitespace: false });
    await waitFor(() => expect(undelivered()).toHaveTextContent(/^abcd$/, { normalizeWhitespace: false }));
  });

  it("shows a query set from outside, such as the URL", async () => {
    const user = userEvent.setup();
    render(<Harness initial="agent:triage" />);
    expect(box()).toHaveTextContent("agent:triage");
    await user.click(screen.getByRole("button", { name: "Load saved query" }));
    expect(box()).toHaveTextContent(/^status:ok$/, { normalizeWhitespace: false });
  });
});
