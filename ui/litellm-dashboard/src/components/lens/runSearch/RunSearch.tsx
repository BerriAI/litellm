"use client";

import "prosemirror-view/style/prosemirror.css";

import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { ProseMirror, ProseMirrorDoc, reactKeys, useEditorEventCallback } from "@handlewithcare/react-prosemirror";
import {
  Bot,
  Box,
  Braces,
  CircleDashed,
  CornerDownLeft,
  Hash,
  type LucideIcon,
  Search,
  SquareChevronRight,
} from "lucide-react";
import { Schema } from "prosemirror-model";
import { EditorState, TextSelection, type Transaction } from "prosemirror-state";
import { Decoration, DecorationSet, type EditorView } from "prosemirror-view";
import { useId, useMemo, useState } from "react";

import type { TraceSummary } from "@/components/view_logs/TraceView/traceTypes";

import { parseRunQuery, type RunField } from "./runQuery";
import { suggest, type Suggestion, type SuggestionMenu } from "./suggestions";

const schema = new Schema({ nodes: { doc: { content: "text*" }, text: {} } });

const createState = (text: string) =>
  EditorState.create({ schema, doc: schema.node("doc", null, text ? schema.text(text) : []), plugins: [reactKeys()] });

const EMIT_WAIT_MS = 150;
const KEEP_MENU_CLOSED = "runSearchKeepMenuClosed";

const FIELD_ICONS: Record<RunField, LucideIcon> = {
  name: SquareChevronRight,
  agent: Bot,
  status: CircleDashed,
  model: Box,
  input: Braces,
  trace_id: Hash,
};

const OPERATORS = [
  { label: "equals", example: "foo:bar" },
  { label: "not equals", example: "-foo:bar" },
  { label: "wildcard match", example: "foo:*bar*" },
  { label: "does not contain", example: "-foo:*bar*" },
];

function highlight(state: EditorState): DecorationSet {
  const marks = parseRunQuery(state.doc.textContent).flatMap((clause) =>
    clause.kind === "field"
      ? [
          Decoration.inline(clause.from, clause.keyTo, { class: "text-info" }),
          Decoration.inline(clause.keyTo, clause.keyTo + 1, { class: "text-muted-foreground" }),
        ]
      : [],
  );
  return DecorationSet.create(state.doc, marks);
}

function applySuggestion(view: EditorView, item: Suggestion) {
  const tr = view.state.tr.insertText(item.insert, item.from, item.to);
  tr.setSelection(TextSelection.create(tr.doc, item.from + item.insert.length));
  if (item.completesClause) tr.setMeta(KEEP_MENU_CLOSED, true);
  view.dispatch(tr);
}

interface RunSearchProps {
  value: string;
  onChange: (value: string) => void;
  /** Loaded runs, the source of value suggestions. */
  runs: readonly TraceSummary[];
}

/** One query box for the runs list: free text plus `key:value` filters, with autocomplete for both. */
export function RunSearch({ value, onChange, runs }: RunSearchProps) {
  const listId = useId();
  const [state, setState] = useState(() => createState(value));
  const [seenValue, setSeenValue] = useState(value);
  const [unechoed, setUnechoed] = useState<readonly string[]>([]);
  const [focused, setFocused] = useState(false);
  const [menuOpen, setMenuOpen] = useState(true);
  const [active, setActive] = useState(0);

  // `value` can trail the editor (the URL updates are throttled), so an echo of text we emitted is not an edit.
  if (value !== seenValue) {
    setSeenValue(value);
    const echo = unechoed.indexOf(value);
    if (echo >= 0) setUnechoed(unechoed.slice(echo + 1));
    else if (value !== state.doc.textContent) {
      setUnechoed([]);
      setState(createState(value));
    }
  }

  const { selection } = state;
  const text = state.doc.textContent;
  const menu = useMemo(() => (selection.empty ? suggest(text, selection.head, runs) : null), [text, selection, runs]);
  const items = menu?.groups.flatMap((group) => group.items) ?? [];
  const shownMenu = focused && menuOpen ? menu : null;
  const visible = shownMenu !== null;
  const activeItem = visible ? items[Math.min(active, items.length - 1)] : undefined;

  const emit = useDebouncedCallback(
    (text: string) => {
      setUnechoed((current) => [...current, text]);
      onChange(text);
    },
    { wait: EMIT_WAIT_MS },
  );

  const dispatchTransaction = (tr: Transaction) => {
    setState((current) => current.apply(tr));
    if (tr.selectionSet || tr.docChanged) setActive(0);
    if (!tr.docChanged) return;
    setMenuOpen(!tr.getMeta(KEEP_MENU_CLOSED));
    emit(tr.doc.textContent);
  };

  const handleKeyDown = (view: EditorView, event: KeyboardEvent): boolean => {
    if (event.key === "Escape" && visible) {
      setMenuOpen(false);
      return true;
    }
    if (event.key === "ArrowDown" && !visible) {
      setMenuOpen(true);
      return true;
    }
    if ((event.key === "ArrowDown" || event.key === "ArrowUp") && items.length > 0) {
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((Math.min(active, items.length - 1) + step + items.length) % items.length);
      return true;
    }
    if ((event.key === "Enter" || event.key === "Tab") && activeItem) {
      applySuggestion(view, activeItem);
      return true;
    }
    return false;
  };

  return (
    <ProseMirror
      state={state}
      dispatchTransaction={dispatchTransaction}
      decorations={highlight}
      handleKeyDown={handleKeyDown}
      handleDOMEvents={{
        focus: () => {
          setFocused(true);
          setMenuOpen(true);
          return false;
        },
        blur: () => {
          setFocused(false);
          return false;
        },
      }}
      transformPastedText={(pasted) => pasted.replace(/\s+/g, " ")}
      attributes={{
        role: "combobox",
        "aria-label": "Search runs",
        "aria-autocomplete": "list",
        "aria-expanded": String(visible),
        "aria-controls": listId,
        ...(activeItem && { "aria-activedescendant": `${listId}-${activeItem.id}` }),
        spellcheck: "false",
        class: "min-w-0 flex-1 overflow-x-auto py-1.5 font-mono text-xs whitespace-pre outline-none",
      }}
    >
      <div className="relative w-full flex-1">
        <div className="flex h-8 items-center gap-2 rounded-md border border-input bg-transparent px-2.5 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 dark:bg-input/30">
          <Search className="size-3.5 shrink-0 text-muted-foreground" />
          <div className="relative flex min-w-0 flex-1">
            <ProseMirrorDoc />
            {!text && (
              <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center font-mono text-xs text-muted-foreground">
                Search runs, or filter like agent:researcher status:error
              </span>
            )}
          </div>
        </div>
        {shownMenu && <SuggestionPanel id={listId} menu={shownMenu} activeId={activeItem?.id} />}
      </div>
    </ProseMirror>
  );
}

function SuggestionPanel({ id, menu, activeId }: { id: string; menu: SuggestionMenu; activeId: string | undefined }) {
  const pick = useEditorEventCallback((view, item: Suggestion) => applySuggestion(view, item));
  return (
    <div className="absolute top-full left-0 z-floating mt-1 w-full max-w-xl overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md">
      <div id={id} role="listbox" aria-label="Search suggestions" className="max-h-80 overflow-y-auto p-1">
        {menu.groups.map((group) => (
          <div key={group.heading} role="group" aria-label={group.heading}>
            <div className="px-2 pt-2 pb-1 text-xs text-muted-foreground">{group.heading}</div>
            {group.items.map((item) => {
              const Icon = FIELD_ICONS[item.field];
              return (
                <div
                  key={item.id}
                  id={`${id}-${item.id}`}
                  role="option"
                  aria-selected={item.id === activeId}
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => pick(item)}
                  className="flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 font-mono text-xs aria-selected:bg-accent aria-selected:text-accent-foreground"
                >
                  <Icon className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate">{item.label}</span>
                </div>
              );
            })}
          </div>
        ))}
        {menu.showOperators && <OperatorHints />}
      </div>
      <div className="flex items-center gap-3 border-t border-border px-3 py-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd>
          Navigate
        </span>
        <span className="flex items-center gap-1">
          <Kbd>
            <CornerDownLeft className="size-3" />
          </Kbd>
          Select
        </span>
      </div>
    </div>
  );
}

function OperatorHints() {
  return (
    <div className="pb-1">
      <div className="px-2 pt-2 pb-1 text-xs text-muted-foreground">Comparison operators</div>
      {OPERATORS.map((op) => (
        <div key={op.label} className="flex items-center justify-between px-2 py-1 font-mono text-xs">
          <span className="text-warning">{op.label}</span>
          <code className="rounded bg-muted px-1.5 text-muted-foreground">{op.example}</code>
        </div>
      ))}
    </div>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border px-1 font-sans">
      {children}
    </kbd>
  );
}
