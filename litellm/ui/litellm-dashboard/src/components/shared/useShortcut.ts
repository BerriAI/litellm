import { type HotkeyCallback, useHotkeys } from "react-hotkeys-hook";

const KEY_OWNERS = "[role='dialog'], [role='menu'], [role='listbox'], [role='combobox']";
const PANE_KEY_OWNERS = `${KEY_OWNERS}, [role='tablist'], [role='separator']`;
const CAPTURE = { capture: true } as const;

/**
 * Where a shortcut lives decides whose keys it yields to.
 * panel: a side panel over the page; yields to dialogs, menus and listboxes opened on top of it.
 * pane: a pane inside a panel; runs first and claims the key so the panel does not also act on it.
 * modal: a dialog that owns every key while open.
 */
export type ShortcutLayer = "panel" | "pane" | "modal";

export interface ShortcutOptions {
  readonly description: string;
  readonly enabled?: boolean;
  readonly layer?: ShortcutLayer;
}

const ownedElsewhere =
  (selector: string | null) =>
  (event: KeyboardEvent): boolean => {
    if (event.defaultPrevented || selector === null) return event.defaultPrevented;
    const insideOwner = event.target instanceof Element && event.target.closest(selector) !== null;
    return insideOwner;
  };

const IGNORES: Readonly<Record<ShortcutLayer, (event: KeyboardEvent) => boolean>> = {
  panel: ownedElsewhere(KEY_OWNERS),
  pane: ownedElsewhere(PANE_KEY_OWNERS),
  modal: ownedElsewhere(null),
};

/** One plain-key shortcut. Typing targets and modified presses such as Cmd+J never trigger it. */
export function useShortcut(
  key: string,
  run: HotkeyCallback,
  { description, enabled = true, layer = "panel" }: ShortcutOptions,
): void {
  const options = {
    description,
    enabled,
    useKey: true,
    preventDefault: true,
    ignoreEventWhen: IGNORES[layer],
    eventListenerOptions: layer === "pane" ? CAPTURE : undefined,
    metadata: { layer },
  };
  useHotkeys(key, run, options);
}
