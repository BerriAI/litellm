import { render, screen } from "@testing-library/react";
import { HotkeysProvider } from "react-hotkeys-hook";
import { describe, expect, it, vi } from "vitest";

import { ShortcutHints } from "./ShortcutHints";
import { BoundShortcut as Bound } from "./__fixtures__/BoundShortcut";

const hintText = () => screen.getByLabelText("Keyboard shortcuts").textContent;

describe("ShortcutHints", () => {
  it("lists what is bound, merging keys that share a description and letting the pane own a shared key", () => {
    const onPress = vi.fn();
    const { rerender } = render(
      <HotkeysProvider>
        <Bound id="j" keys={["j", "trace"]} layer="panel" onPress={onPress} />
        <Bound id="k" keys={["k", "trace"]} layer="panel" onPress={onPress} />
        <Bound id="esc" keys={["escape", "close trace"]} layer="panel" onPress={onPress} />
        <Bound id="down" keys={["down", "step"]} layer="pane" onPress={onPress} />
        <Bound id="up" keys={["up", "step"]} layer="pane" onPress={onPress} />
        <Bound id="paneEsc" keys={["escape", "close pane"]} layer="pane" onPress={onPress} />
        <ShortcutHints />
      </HotkeysProvider>,
    );
    expect(hintText()).toBe("↑/↓ stepJ/K traceEsc close pane");

    rerender(
      <HotkeysProvider>
        <Bound id="j" keys={["j", "trace"]} layer="panel" onPress={onPress} />
        <Bound id="k" keys={["k", "trace"]} layer="panel" onPress={onPress} />
        <Bound id="esc" keys={["escape", "close trace"]} layer="panel" onPress={onPress} />
        <ShortcutHints />
      </HotkeysProvider>,
    );
    expect(hintText()).toBe("J/K traceEsc close trace");
  });

  it("renders nothing while no shortcut is bound", () => {
    render(
      <HotkeysProvider>
        <Bound id="off" keys={["j", "trace"]} layer="panel" enabled={false} onPress={vi.fn()} />
        <ShortcutHints />
      </HotkeysProvider>,
    );
    expect(screen.queryByLabelText("Keyboard shortcuts")).not.toBeInTheDocument();
  });
});
