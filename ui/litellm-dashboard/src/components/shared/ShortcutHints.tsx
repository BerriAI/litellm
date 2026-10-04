"use client";

import { Fragment, type ReactNode } from "react";
import { useHotkeysContext } from "react-hotkeys-hook";

import { cn } from "@/lib/cva.config";

type BoundShortcut = ReturnType<typeof useHotkeysContext>["hotkeys"][number];

export interface ShortcutHint {
  readonly keys: readonly string[];
  readonly description: string;
}

const KEY_LABELS: Readonly<Record<string, string>> = {
  arrowup: "↑",
  arrowdown: "↓",
  arrowleft: "←",
  arrowright: "→",
  escape: "Esc",
};
const ARROWS: readonly string[] = ["arrowup", "arrowdown", "arrowleft", "arrowright"];

export const keyLabel = (key: string): string => KEY_LABELS[key] ?? key.toUpperCase();

const keyRank = (key: string): number => {
  const arrow = ARROWS.indexOf(key);
  if (arrow >= 0) return arrow;
  return key === "escape" ? Number.MAX_SAFE_INTEGER : ARROWS.length + key.charCodeAt(0);
};

const plainKey = (shortcut: BoundShortcut): string | null => {
  const single = Boolean(shortcut.description) && shortcut.keys?.length === 1;
  const modified = [shortcut.alt, shortcut.ctrl, shortcut.meta, shortcut.shift].some(Boolean);
  return single && !modified ? shortcut.keys?.[0] ?? null : null;
};

const innermost = (current: BoundShortcut, next: BoundShortcut): BoundShortcut =>
  next.metadata?.layer === "pane" ? next : current;

const mergeAdjacent = (hints: readonly ShortcutHint[], key: string, description: string): readonly ShortcutHint[] => {
  const last = hints.at(-1);
  if (last?.description !== description) return [...hints, { keys: [key], description }];
  return [...hints.slice(0, -1), { keys: [...last.keys, key], description }];
};

/** One hint per key with the innermost binding winning; keys that share a description read as one hint. */
export function shortcutHints(shortcuts: readonly BoundShortcut[]): readonly ShortcutHint[] {
  const winners = shortcuts.reduce((byKey, shortcut) => {
    const key = plainKey(shortcut);
    if (key === null) return byKey;
    const current = byKey.get(key);
    return new Map(byKey).set(key, current ? innermost(current, shortcut) : shortcut);
  }, new Map<string, BoundShortcut>());
  return [...winners.entries()]
    .sort(([a], [b]) => keyRank(a) - keyRank(b))
    .reduce<
      readonly ShortcutHint[]
    >((hints, [key, shortcut]) => mergeAdjacent(hints, key, shortcut.description ?? ""), []);
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-sm border border-border border-b-2 bg-muted px-[3px] font-mono text-muted-foreground">
      {children}
    </kbd>
  );
}

/** The shortcuts active right now, read from what is bound rather than typed by hand. */
export function ShortcutHints({ className }: { className?: string }) {
  const { hotkeys } = useHotkeysContext();
  const hints = shortcutHints(hotkeys);
  if (hints.length === 0) return null;
  return (
    <div
      aria-label="Keyboard shortcuts"
      data-slot="shortcut-hints"
      className={cn("flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground", className)}
    >
      {hints.map((hint) => (
        <span key={hint.keys.join("/")} className="whitespace-nowrap">
          {hint.keys.map((key, index) => (
            <Fragment key={key}>
              {index > 0 && "/"}
              <Kbd>{keyLabel(key)}</Kbd>
            </Fragment>
          ))}{" "}
          {hint.description}
        </span>
      ))}
    </div>
  );
}
