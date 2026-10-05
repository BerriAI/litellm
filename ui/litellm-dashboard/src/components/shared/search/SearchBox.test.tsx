import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useEffect, useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { NOTE_INDEX, type NoteField, NOTE_QUERY, notes } from "./__fixtures__/notes";
import { SearchBox } from "./SearchBox";
import type { SearchQuery } from "./searchQuery";
import { itemValues, type ValueSource } from "./valueSource";

/** A toy API: each filter becomes a flag, so the test can read the translation off the footer. */
function notesCli(query: SearchQuery<string>): string {
  const flags = [...query.text, ...query.filters.map((f) => `--${f.field}${f.op === "neq" ? "!" : ""}=${f.value}`)];
  return `notes ${flags.join(" ")}`.trim();
}

function NoteSearchWithHint({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <SearchBox.Root
      language={NOTE_QUERY}
      values={itemValues(NOTE_INDEX, notes)}
      value={value}
      onValueChange={onChange}
      label="Search notes"
    >
      <SearchBox.Input placeholder="Search notes" />
      <SearchBox.Suggestions>
        <SearchBox.CopyCommand title="The notes CLI takes the same flags" command={notesCli} />
      </SearchBox.Suggestions>
    </SearchBox.Root>
  );
}

function NoteSearch({
  value,
  onChange,
  values = itemValues(NOTE_INDEX, notes),
}: {
  value: string;
  onChange: (value: string) => void;
  values?: ValueSource<NoteField>;
}) {
  return (
    <SearchBox.Root language={NOTE_QUERY} values={values} value={value} onValueChange={onChange} label="Search notes">
      <SearchBox.Input placeholder="Search notes" />
      <SearchBox.Suggestions />
    </SearchBox.Root>
  );
}

function Harness({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <NoteSearch value={value} onChange={setValue} />
      <button type="button" onClick={() => setValue("tag:cron")}>
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
      <NoteSearch value={value} onChange={(next) => setQueue((q) => [...q, next])} />
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

const box = () => screen.getByRole("combobox", { name: "Search notes" });
const undelivered = () => screen.getByRole("status", { name: "undelivered" });
const listbox = () => screen.getByRole("listbox", { name: "Search suggestions" });

describe("SearchBox", () => {
  it("builds a filter from the keyboard: field, then value", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    expect(screen.getByText("Search notes")).toBeVisible();
    await user.click(box());
    expect(box()).toHaveAttribute("aria-expanded", "true");
    await user.keyboard("-ta");
    expect(screen.queryByText("Search notes")).not.toBeInTheDocument();
    expect(
      within(listbox())
        .getAllByRole("option")
        .map((o) => o.textContent),
    ).toEqual(["tag"]);
    await user.keyboard("{Enter}");
    expect(box()).toHaveTextContent(/^-tag:$/, { normalizeWhitespace: false });
    expect(within(listbox()).getByRole("group", { name: "tag" })).toBeVisible();
    expect(within(listbox()).getByText("does not contain")).toBeVisible();

    await user.keyboard("{ArrowDown}");
    const cron = within(listbox()).getByRole("option", { name: "cron" });
    expect(cron).toHaveAttribute("aria-selected", "true");
    expect(box()).toHaveAttribute("aria-activedescendant", cron.id);
    await user.keyboard("{Enter}");
    expect(box()).toHaveTextContent(/^-tag:cron $/, {
      normalizeWhitespace: false,
    });
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(box()).toHaveAttribute("aria-expanded", "false");
  });

  it("wraps arrow navigation and picks with Tab", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    await user.keyboard("{ArrowUp}");
    expect(within(listbox()).getByRole("option", { name: "id" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{ArrowDown}{ArrowDown}{Tab}");
    expect(box()).toHaveTextContent(/^tag:$/, { normalizeWhitespace: false });
    await user.keyboard("{ArrowDown}{ArrowDown}e");
    expect(within(listbox()).getByRole("option", { name: "researcher" })).toHaveAttribute("aria-selected", "true");
  });

  it("picks a suggestion with the mouse without losing focus", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(box());
    await user.keyboard("tag:");
    await user.click(within(listbox()).getByRole("option", { name: "researcher" }));
    expect(box()).toHaveTextContent(/^tag:researcher $/, {
      normalizeWhitespace: false,
    });
    expect(box()).toHaveFocus();
    await user.keyboard("refund");
    expect(box()).toHaveTextContent(/^tag:researcher refund$/, {
      normalizeWhitespace: false,
    });
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
    render(<NoteSearch value="" onChange={onChange} />);
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
    await waitFor(() =>
      expect(undelivered()).toHaveTextContent(/^ab$/, {
        normalizeWhitespace: false,
      }),
    );
    await user.keyboard("c");
    await user.click(screen.getByRole("button", { name: "Deliver next" }));
    await user.click(box());
    await user.keyboard("d");
    expect(box()).toHaveTextContent(/^abcd$/, { normalizeWhitespace: false });
    await waitFor(() =>
      expect(undelivered()).toHaveTextContent(/^abcd$/, {
        normalizeWhitespace: false,
      }),
    );
  });

  it("shows a query set from outside, such as the URL", async () => {
    const user = userEvent.setup();
    render(<Harness initial="tag:triage" />);
    expect(box()).toHaveTextContent("tag:triage");
    await user.click(screen.getByRole("button", { name: "Load saved query" }));
    expect(box()).toHaveTextContent(/^tag:cron$/, {
      normalizeWhitespace: false,
    });
  });

  it("asks the value source for the field and prefix at the cursor, showing a loading row until it answers", async () => {
    const user = userEvent.setup();
    const asked: string[] = [];
    const facets: ValueSource<NoteField> = {
      useValues(field, prefix) {
        const [answer, setAnswer] = useState<readonly string[] | null>(null);
        useEffect(() => {
          if (field === null) return;
          asked.push(`${field}:${prefix}`);
          setAnswer(null);
          const timer = setTimeout(() => setAnswer(["remote-researcher", "remote-review"]), 20);
          return () => clearTimeout(timer);
        }, [field, prefix]);
        if (field === null) return { values: [], loading: false };
        return answer ? { values: answer, loading: false } : { values: [], loading: true };
      },
    };
    render(<NoteSearch value="" onChange={vi.fn()} values={facets} />);
    await user.click(box());
    await user.keyboard("body:x tag:re");
    expect(within(listbox()).getByRole("status")).toHaveTextContent("Loading values…");
    const option = await within(listbox()).findByRole("option", {
      name: "remote-researcher",
    });
    expect(within(listbox()).queryByRole("status")).not.toBeInTheDocument();
    await user.click(option);
    expect(box()).toHaveTextContent(/^body:x tag:remote-researcher $/, {
      normalizeWhitespace: false,
    });
    expect(asked).toEqual(["tag:", "tag:r", "tag:re"]);
  });

  it("copies the command for the current query from the footer and keeps the box open", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<NoteSearchWithHint value="" onChange={onChange} />);
    expect(screen.queryByRole("button", { name: "Copy as curl" })).not.toBeInTheDocument();
    await user.click(box());
    expect(screen.getByTitle("The notes CLI takes the same flags")).toBeVisible();
    await user.keyboard("-tag:cr");
    await user.click(screen.getByRole("button", { name: "Copy as curl" }));
    expect(await navigator.clipboard.readText()).toBe("notes --tag!=cr");
    expect(screen.getByRole("button", { name: "Copied" })).toBeVisible();
    expect(box()).toHaveAttribute("aria-expanded", "true");
    await user.keyboard("o");
    expect(screen.getByRole("button", { name: "Copy as curl" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy as curl" }));
    expect(await navigator.clipboard.readText()).toBe("notes --tag!=cro");
  });
});
