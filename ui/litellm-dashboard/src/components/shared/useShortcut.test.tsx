import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HotkeysProvider } from "react-hotkeys-hook";
import { describe, expect, it, vi } from "vitest";

import { BoundShortcut as Bound } from "./__fixtures__/BoundShortcut";

describe("useShortcut", () => {
  it("lets a pane claim a key so the panel around it does not also act on it", () => {
    const onPress = vi.fn();
    render(
      <HotkeysProvider>
        <Bound id="panel" keys={["escape", "close"]} layer="panel" onPress={onPress} />
        <Bound id="pane" keys={["escape", "close"]} layer="pane" onPress={onPress} />
      </HotkeysProvider>,
    );
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onPress.mock.calls).toEqual([["pane"]]);
  });

  it("falls back to the panel binding once the pane's shortcut is disabled", () => {
    const onPress = vi.fn();
    render(
      <HotkeysProvider>
        <Bound id="panel" keys={["escape", "close"]} layer="panel" onPress={onPress} />
        <Bound id="pane" keys={["escape", "close"]} layer="pane" enabled={false} onPress={onPress} />
      </HotkeysProvider>,
    );
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(onPress.mock.calls).toEqual([["panel"]]);
  });

  it("yields to typing, modified presses, and widgets that own their keys", async () => {
    const user = userEvent.setup();
    const onPress = vi.fn();
    render(
      <HotkeysProvider>
        <input aria-label="search" />
        <div role="menu">
          <button type="button">menu item</button>
        </div>
        <div role="tablist">
          <button type="button" role="tab">
            tab
          </button>
        </div>
        <Bound id="panel" keys={["j", "trace"]} layer="panel" onPress={onPress} />
        <Bound id="pane" keys={["right", "fold"]} layer="pane" onPress={onPress} />
      </HotkeysProvider>,
    );
    fireEvent.keyDown(screen.getByLabelText("search"), { key: "j" });
    fireEvent.keyDown(screen.getByText("menu item"), { key: "j" });
    fireEvent.keyDown(screen.getByRole("tab"), { key: "ArrowRight" });
    await user.keyboard("{Meta>}j{/Meta}{Control>}j{/Control}{Shift>}j{/Shift}");
    expect(onPress).not.toHaveBeenCalled();
    fireEvent.keyDown(screen.getByRole("tab"), { key: "j" });
    await user.keyboard("{ArrowRight}");
    expect(onPress.mock.calls).toEqual([["panel"], ["pane"]]);
  });

  it("keeps working for a modal while focus sits inside its dialog", () => {
    const onPress = vi.fn();
    render(
      <HotkeysProvider>
        <div role="dialog">
          <button type="button">inside</button>
        </div>
        <Bound id="modal" keys={["j", "log"]} layer="modal" onPress={onPress} />
        <Bound id="panel" keys={["k", "trace"]} layer="panel" onPress={onPress} />
      </HotkeysProvider>,
    );
    fireEvent.keyDown(screen.getByText("inside"), { key: "j" });
    fireEvent.keyDown(screen.getByText("inside"), { key: "k" });
    expect(onPress.mock.calls).toEqual([["modal"]]);
  });
});
