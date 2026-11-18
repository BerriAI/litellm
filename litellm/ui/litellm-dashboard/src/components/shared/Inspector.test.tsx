import { fireEvent, render, screen, within } from "@testing-library/react";
import { HotkeysProvider } from "react-hotkeys-hook";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { clampPanelWidth, Inspector } from "./Inspector";

interface Note {
  id: string;
  title: string;
}

const notes: Note[] = [
  { id: "a", title: "first" },
  { id: "b", title: "second" },
  { id: "c", title: "third" },
];
const noteKey = (note: Note) => note.id;
const STORAGE_KEY = "test.inspector.width";

function Notes({
  initial = null,
  items = notes,
  onSelectedChange,
  fullScreen,
  onFullScreenChange,
  extra,
}: {
  initial?: Note | null;
  items?: readonly Note[];
  onSelectedChange?: (note: Note | null) => void;
  fullScreen?: boolean;
  onFullScreenChange?: (fullScreen: boolean) => void;
  extra?: React.ReactNode;
}) {
  const [selected, setSelected] = useState<Note | null>(initial);
  return (
    <HotkeysProvider>
      {extra}
      <Inspector.Root
        items={items}
        itemKey={noteKey}
        selected={selected}
        onSelectedChange={(note) => {
          setSelected(note);
          onSelectedChange?.(note);
        }}
        noun="note"
        storageKey={STORAGE_KEY}
        fullScreen={fullScreen}
        onFullScreenChange={onFullScreenChange}
      >
        <table>
          <tbody>
            {items.map((note) => (
              <Inspector.Row key={note.id} item={note} render={<tr tabIndex={0} />}>
                <td>{note.title}</td>
              </Inspector.Row>
            ))}
          </tbody>
        </table>
        <Inspector.Panel label="Note details" testId="note-panel">
          {(note: Note) => <div data-testid="note-body">note {note.title}</div>}
        </Inspector.Panel>
      </Inspector.Root>
    </HotkeysProvider>
  );
}

const panel = () => screen.getByRole("complementary", { name: "Note details" });
const row = (title: string) => screen.getByRole("row", { name: title });

const mockReducedMotion = (reduce: boolean) =>
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query: string) =>
      ({
        matches: reduce && query.includes("reduce"),
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }) as unknown as MediaQueryList,
  );

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("clampPanelWidth", () => {
  it("keeps at least 700px and a 100px strip of list on wide screens", () => {
    expect(clampPanelWidth(100, 1600)).toBe(700);
    expect(clampPanelWidth(2000, 1600)).toBe(1500);
    expect(clampPanelWidth(900, 1600)).toBe(900);
  });

  it("never exceeds the screen on viewports narrower than the minimum width", () => {
    expect(clampPanelWidth(900, 600)).toBe(600);
    expect(clampPanelWidth(100, 600)).toBe(600);
  });
});

describe("Inspector", () => {
  it("opens an item from its row, marks the row selected, and toggles it closed on a second press", () => {
    render(<Notes />);
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
    expect(row("second")).toHaveAttribute("data-state", "idle");

    fireEvent.click(row("second"));
    expect(screen.getByTestId("note-body")).toHaveTextContent("note second");
    expect(row("second")).toHaveAttribute("aria-selected", "true");
    expect(row("second")).toHaveAttribute("data-state", "selected");
    expect(row("second")).toHaveAttribute("data-slot", "inspector-row");
    expect(row("first")).toHaveAttribute("aria-selected", "false");

    fireEvent.click(row("second"));
    expect(row("second")).toHaveAttribute("data-state", "idle");
  });

  it("opens a focused row with Enter or Space but not from keys inside it", () => {
    mockReducedMotion(true);
    render(<Notes />);
    fireEvent.keyDown(row("third"), { key: "Enter" });
    expect(screen.getByTestId("note-body")).toHaveTextContent("note third");
    fireEvent.keyDown(within(row("third")).getByText("third"), { key: " " });
    expect(screen.getByTestId("note-body")).toHaveTextContent("note third");
    fireEvent.keyDown(row("third"), { key: " " });
    expect(screen.queryByTestId("note-body")).not.toBeInTheDocument();
  });

  it("steps with J / K and the header arrows, showing the position and disabling at the ends", () => {
    mockReducedMotion(true);
    render(<Notes initial={notes[0]} />);
    expect(panel()).toHaveTextContent("1 / 3");
    expect(screen.getByRole("button", { name: "Previous note (K)" })).toBeDisabled();

    fireEvent.keyDown(document.body, { key: "j" });
    expect(screen.getByTestId("note-body")).toHaveTextContent("note second");
    fireEvent.click(screen.getByRole("button", { name: "Next note (J)" }));
    expect(panel()).toHaveTextContent("3 / 3");
    expect(screen.getByRole("button", { name: "Next note (J)" })).toBeDisabled();
    fireEvent.keyDown(document.body, { key: "k" });
    expect(screen.getByTestId("note-body")).toHaveTextContent("note second");

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });
  it("lists the active shortcuts in the panel footer", () => {
    mockReducedMotion(true);
    render(<Notes initial={notes[0]} />);
    expect(within(panel()).getByLabelText("Keyboard shortcuts")).toHaveTextContent("J/K noteEsc close");
  });

  it("shows an item that is not in the list and steps from the top of the list", () => {
    mockReducedMotion(true);
    render(<Notes initial={{ id: "z", title: "elsewhere" }} />);
    expect(screen.getByTestId("note-body")).toHaveTextContent("note elsewhere");
    expect(panel()).not.toHaveTextContent("/ 3");
    expect(screen.getByRole("button", { name: "Previous note (K)" })).toBeDisabled();
    fireEvent.keyDown(document.body, { key: "j" });
    expect(screen.getByTestId("note-body")).toHaveTextContent("note first");
  });

  it("leaves J, K and Escape to an open menu or listbox", () => {
    const onSelectedChange = vi.fn();
    render(
      <Notes
        initial={notes[1]}
        onSelectedChange={onSelectedChange}
        extra={
          <>
            <div role="menu">
              <button type="button">menu item</button>
            </div>
            <div role="listbox">
              <button type="button">option</button>
            </div>
          </>
        }
      />,
    );
    for (const target of [screen.getByText("menu item"), screen.getByText("option")]) {
      for (const key of ["j", "k", "Escape"]) fireEvent.keyDown(target, { key });
    }
    expect(onSelectedChange).not.toHaveBeenCalled();
    fireEvent.keyDown(document.body, { key: "j" });
    expect(onSelectedChange).toHaveBeenCalledExactlyOnceWith(notes[2]);
  });

  it("closes on a press outside the panel but not inside it, on a row, in a dialog, or on its overlay", () => {
    const onSelectedChange = vi.fn();
    render(
      <Notes
        initial={notes[0]}
        onSelectedChange={onSelectedChange}
        extra={
          <>
            <button type="button">outside</button>
            <div role="dialog">dialog</div>
            <div data-slot="sheet-overlay">overlay</div>
          </>
        }
      />,
    );
    fireEvent.mouseDown(screen.getByTestId("note-body"));
    fireEvent.mouseDown(within(row("second")).getByText("second"));
    fireEvent.mouseDown(screen.getByText("dialog"));
    fireEvent.mouseDown(screen.getByText("overlay"));
    fireEvent.mouseDown(screen.getByText("outside"), { button: 2 });
    expect(onSelectedChange).not.toHaveBeenCalled();
    fireEvent.mouseDown(screen.getByText("outside"));
    expect(onSelectedChange).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("remembers a resized width per storage key and restores it after full screen", () => {
    const onFullScreenChange = vi.fn();
    function Controlled() {
      const [fullScreen, setFullScreen] = useState(false);
      return (
        <Notes
          initial={notes[0]}
          fullScreen={fullScreen}
          onFullScreenChange={(next) => {
            setFullScreen(next);
            onFullScreenChange(next);
          }}
        />
      );
    }
    render(<Controlled />);
    fireEvent.keyDown(screen.getByRole("separator", { name: "Resize note panel" }), { key: "ArrowLeft" });
    const resizedWidth = panel().style.width;
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe(String(Number.parseInt(resizedWidth, 10)));

    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    expect(onFullScreenChange).toHaveBeenLastCalledWith(true);
    expect(panel()).toHaveStyle({ width: "100%" });
    expect(screen.queryByRole("separator", { name: "Resize note panel" })).not.toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "j" });
    expect(panel()).toHaveStyle({ width: "100%" });

    fireEvent.click(screen.getByRole("button", { name: "Exit full screen" }));
    expect(panel()).toHaveStyle({ width: resizedWidth });
  });

  it("manages full screen itself when the parent does not", () => {
    mockReducedMotion(true);
    render(<Notes initial={notes[0]} />);
    fireEvent.click(screen.getByRole("button", { name: "Enter full screen" }));
    expect(panel()).toHaveStyle({ width: "100%" });
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });

  it("hands J, K and Escape to an open inspector nested inside its panel and takes them back when it closes", () => {
    mockReducedMotion(true);
    const outer = vi.fn();
    const inner = vi.fn();
    function Nested() {
      const [innerNote, setInnerNote] = useState<Note | null>(notes[0]);
      return (
        <HotkeysProvider>
          <Inspector.Root
            items={notes}
            itemKey={noteKey}
            selected={notes[1]}
            onSelectedChange={outer}
            noun="note"
            storageKey="outer"
          >
            <Inspector.Panel label="Outer">
              {() => (
                <Inspector.Root
                  items={notes}
                  itemKey={noteKey}
                  selected={innerNote}
                  onSelectedChange={(note) => {
                    inner(note);
                    setInnerNote(note);
                  }}
                  noun="reply"
                  storageKey="inner"
                >
                  <Inspector.Panel label="Inner">{(note: Note) => <span>reply {note.title}</span>}</Inspector.Panel>
                </Inspector.Root>
              )}
            </Inspector.Panel>
          </Inspector.Root>
        </HotkeysProvider>
      );
    }
    render(<Nested />);
    fireEvent.keyDown(document.body, { key: "j" });
    expect(inner).toHaveBeenLastCalledWith(notes[1]);
    expect(outer).not.toHaveBeenCalled();

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(inner).toHaveBeenLastCalledWith(null);
    expect(screen.queryByRole("complementary", { name: "Inner" })).not.toBeInTheDocument();
    expect(outer).not.toHaveBeenCalled();

    fireEvent.keyDown(document.body, { key: "k" });
    expect(outer).toHaveBeenLastCalledWith(notes[0]);
  });

  it("unmounts right away on close when the user prefers reduced motion", () => {
    mockReducedMotion(true);
    render(<Notes initial={notes[0]} />);
    fireEvent.click(screen.getByRole("button", { name: "Close note (Esc)" }));
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });

  it("plays the exit animation before unmounting when motion is allowed", () => {
    mockReducedMotion(false);
    render(<Notes initial={notes[0]} />);
    fireEvent.click(screen.getByRole("button", { name: "Close (Esc)" }));
    expect(panel()).toHaveClass("animate-trace-drawer-out");
    expect(panel()).toHaveAttribute("data-state", "closing");
    fireEvent.animationEnd(panel());
    expect(screen.queryByRole("complementary")).not.toBeInTheDocument();
  });
});
