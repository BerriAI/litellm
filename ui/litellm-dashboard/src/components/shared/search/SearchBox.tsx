"use client";

import "prosemirror-view/style/prosemirror.css";

import { useDebouncer } from "@tanstack/react-pacer/debouncer";
import { ProseMirror, ProseMirrorDoc, reactKeys, useEditorEventCallback } from "@handlewithcare/react-prosemirror";
import { Check, Copy, CornerDownLeft, Loader2, type LucideIcon, Search } from "lucide-react";
import { Schema } from "prosemirror-model";
import { EditorState, Plugin, TextSelection, type Transaction } from "prosemirror-state";
import { Decoration, DecorationSet, type EditorView } from "prosemirror-view";
import { type ComponentProps, createContext, type ReactNode, useContext, useId, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { type FilterOp, languageOps, parseQuery, type QueryClause, type QueryLanguage } from "./language";
import { type SearchQuery, toSearchQuery } from "./searchQuery";
import { completingField, completingPrefix, suggest, type Suggestion, type SuggestionMenu } from "./suggestions";
import { NO_VALUES, type ValueSource } from "./valueSource";

const schema = new Schema({ nodes: { doc: { content: "text*" }, text: {} } });

const EMIT_WAIT_MS = 150;
const KEEP_MENU_CLOSED = "searchBoxKeepMenuClosed";

const OPERATORS: readonly { readonly op: FilterOp; readonly label: string; readonly example: string }[] = [
  { op: "eq", label: "equals", example: "foo:bar" },
  { op: "neq", label: "not equals", example: "-foo:bar" },
  { op: "glob", label: "wildcard match", example: "foo:*bar*" },
  { op: "nglob", label: "does not contain", example: "-foo:*bar*" },
];

/** Colours each `key:` so the filters stand out from free text. */
function clauseHighlight<F extends string>(language: QueryLanguage<F>): Plugin {
  return new Plugin({
    props: {
      decorations(state) {
        const marks = parseQuery(language, state.doc.textContent).flatMap((clause) =>
          clause.kind === "field"
            ? [
                Decoration.inline(clause.from, clause.keyTo, {
                  class: "text-info",
                }),
                Decoration.inline(clause.keyTo, clause.keyTo + 1, {
                  class: "text-muted-foreground",
                }),
              ]
            : [],
        );
        return DecorationSet.create(state.doc, marks);
      },
    },
  });
}

const createState = <F extends string>(language: QueryLanguage<F>, text: string) =>
  EditorState.create({
    schema,
    doc: schema.node("doc", null, text ? schema.text(text) : []),
    plugins: [reactKeys(), clauseHighlight(language)],
  });

function applySuggestion(view: EditorView, item: Suggestion<string>) {
  const tr = view.state.tr.insertText(item.insert, item.from, item.to);
  tr.setSelection(TextSelection.create(tr.doc, item.from + item.insert.length));
  if (item.completesClause) tr.setMeta(KEEP_MENU_CLOSED, true);
  view.dispatch(tr);
}

interface SearchBoxState {
  readonly listId: string;
  readonly text: string;
  readonly clauses: readonly QueryClause<string>[];
  /** The menu while it is open, or null when closed. */
  readonly menu: SuggestionMenu<string> | null;
  readonly activeId: string | undefined;
  readonly icon: (field: string) => LucideIcon;
  /** The ops the language can express, so the help only lists forms the surface honors. */
  readonly ops: readonly FilterOp[];
}

const SearchBoxContext = createContext<SearchBoxState | null>(null);

function useSearchBox(): SearchBoxState {
  const state = useContext(SearchBoxContext);
  if (!state) throw new Error("SearchBox parts must be rendered inside SearchBox.Root");
  return state;
}

export type SearchBoxRootProps<F extends string> = Omit<ComponentProps<"div">, "onChange"> & {
  language: QueryLanguage<F>;
  /** Where value suggestions come from: the loaded items, or a server lookup. */
  values: ValueSource<F>;
  value: string;
  onValueChange: (value: string) => void;
  /** Accessible name of the combobox. */
  label: string;
};

/** A one-line query editor: free text plus `key:value` filters. Owns the text, the menu and the keyboard. */
function Root<F extends string>({
  language,
  values,
  value,
  onValueChange,
  label,
  className,
  children,
  ...props
}: SearchBoxRootProps<F>) {
  const listId = useId();
  const [state, setState] = useState(() => createState(language, value));
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
      setState(createState(language, value));
    }
  }

  const { selection } = state;
  const text = state.doc.textContent;
  const clauses = useMemo(() => parseQuery(language, text), [language, text]);
  const completing = selection.empty ? completingField(language, text, selection.head) : null;
  const fetched = values.useValues(completing, completingPrefix(language, text, selection.head));
  const menu = useMemo(() => {
    if (!selection.empty) return null;
    return suggest(language, text, selection.head, (field) => (field === completing ? fetched : NO_VALUES));
  }, [language, text, selection, completing, fetched]);
  const options = menu?.groups.flatMap((group) => group.items) ?? [];
  const shownMenu = focused && menuOpen ? menu : null;
  const visible = shownMenu !== null;
  const activeItem = visible ? options[Math.min(active, options.length - 1)] : undefined;
  const context = useMemo<SearchBoxState>(
    () => ({
      listId,
      text,
      clauses,
      menu: shownMenu,
      activeId: activeItem?.id,
      icon: (field) => language.fields[field as F].icon,
      ops: languageOps(language),
    }),
    [listId, text, clauses, shownMenu, activeItem, language],
  );

  const emitter = useDebouncer(
    (next: string) => {
      setUnechoed((current) => [...current, next]);
      onValueChange(next);
    },
    { wait: EMIT_WAIT_MS },
  );

  const dispatchTransaction = (tr: Transaction) => {
    setState((current) => current.apply(tr));
    if (tr.selectionSet || tr.docChanged) setActive(0);
    if (!tr.docChanged) return;
    setMenuOpen(!tr.getMeta(KEEP_MENU_CLOSED));
    emitter.maybeExecute(tr.doc.textContent);
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
    if ((event.key === "ArrowDown" || event.key === "ArrowUp") && options.length > 0) {
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((Math.min(active, options.length - 1) + step + options.length) % options.length);
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
      handleKeyDown={handleKeyDown}
      handleDOMEvents={{
        focus: () => {
          setFocused(true);
          setMenuOpen(true);
          return false;
        },
        blur: () => {
          setFocused(false);
          emitter.flush();
          return false;
        },
      }}
      transformPastedText={(pasted) => pasted.replace(/\s+/g, " ")}
      attributes={{
        role: "combobox",
        "aria-label": label,
        "aria-autocomplete": "list",
        "aria-expanded": String(visible),
        "aria-controls": listId,
        ...(activeItem && {
          "aria-activedescendant": `${listId}-${activeItem.id}`,
        }),
        spellcheck: "false",
        class: "min-w-0 flex-1 overflow-x-auto py-1.5 font-mono text-xs whitespace-pre outline-none",
      }}
    >
      <SearchBoxContext.Provider value={context}>
        <div data-slot="search-box" className={cn("relative w-full flex-1", className)} {...props}>
          {children}
        </div>
      </SearchBoxContext.Provider>
    </ProseMirror>
  );
}

export type SearchBoxInputProps = ComponentProps<"div"> & {
  placeholder: string;
  /** Results for the current query are loading: the search icon becomes a spinner. */
  busy?: boolean;
};

/** The bordered field holding the editor; shows `placeholder` while the query is empty. */
function Input({ placeholder, busy = false, className, ...props }: SearchBoxInputProps) {
  const { text } = useSearchBox();
  return (
    <div
      data-slot="search-box-input"
      className={cn(
        "flex h-8 items-center gap-2 rounded-md border border-input bg-transparent px-2.5 focus-within:border-ring focus-within:ring-[3px] focus-within:ring-ring/50 dark:bg-input/30",
        className,
      )}
      {...props}
    >
      {busy ? (
        <Loader2
          role="status"
          aria-label="Loading results"
          className="size-3.5 shrink-0 animate-spin text-muted-foreground motion-reduce:animate-none"
        />
      ) : (
        <Search className="size-3.5 shrink-0 text-muted-foreground" />
      )}
      <div className="relative flex min-w-0 flex-1">
        <ProseMirrorDoc />
        {!text && (
          <span className="pointer-events-none absolute inset-0 flex items-center font-mono text-xs text-muted-foreground">
            <span className="truncate">{placeholder}</span>
          </span>
        )}
      </div>
    </div>
  );
}

export type SearchBoxSuggestionsProps = ComponentProps<"div">;

/**
 * The autocomplete listbox under the input: fields, then values, with operator help and key hints.
 * `children` render in the footer beside the key hints, e.g. a `CopyCommand`.
 */
function Suggestions({ className, children, ...props }: SearchBoxSuggestionsProps) {
  const { listId, menu, activeId, icon, ops } = useSearchBox();
  const pick = useEditorEventCallback((view, item: Suggestion<string>) => applySuggestion(view, item));
  if (!menu) return null;
  return (
    <div
      data-slot="search-box-suggestions"
      className={cn(
        "absolute top-full left-0 z-floating mt-1 w-full max-w-xl overflow-hidden rounded-md border border-border bg-popover text-popover-foreground shadow-md",
        className,
      )}
      {...props}
    >
      <div id={listId} role="listbox" aria-label="Search suggestions" className="max-h-80 overflow-y-auto p-1">
        {menu.groups.map((group) => (
          <div key={group.heading} role="group" aria-label={group.heading}>
            <div className="px-2 pt-2 pb-1 text-xs text-muted-foreground">{group.heading}</div>
            {group.items.map((item) => {
              const Icon = icon(item.field);
              return (
                <div
                  key={item.id}
                  id={`${listId}-${item.id}`}
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
        {menu.loading && (
          <div role="status" className="px-2 py-1.5 font-mono text-xs text-muted-foreground">
            Loading values…
          </div>
        )}
        {menu.showOperators && <OperatorHints ops={ops} />}
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
        {children}
      </div>
    </div>
  );
}

export type SearchBoxCopyCommandProps<F extends string> = {
  /** Explains what gets copied; shown on hover. */
  title: string;
  /** The runnable command for the current query. */
  command: (query: SearchQuery<F>) => string;
};

type CopyOutcome = { readonly command: string; readonly outcome: "copied" | "failed" } | null;

const copyLabel = (last: CopyOutcome, command: string): string => {
  if (last?.command !== command) return "Copy as curl";
  return last.outcome === "copied" ? "Copied" : "Copy failed";
};

/** Puts the current query on the clipboard as an API call, from the suggestions footer. */
function CopyCommand<F extends string>({ title, command }: SearchBoxCopyCommandProps<F>) {
  const { clauses } = useSearchBox();
  const [last, setLast] = useState<CopyOutcome>(null);
  const current = useMemo(() => command(toSearchQuery(clauses as readonly QueryClause<F>[])), [command, clauses]);
  const copy = () =>
    navigator.clipboard.writeText(current).then(
      () => setLast({ command: current, outcome: "copied" }),
      () => setLast({ command: current, outcome: "failed" }),
    );
  const label = copyLabel(last, current);
  return (
    <Button
      type="button"
      variant="ghost"
      size="xs"
      title={title}
      data-slot="search-box-copy-command"
      className="ml-auto shrink-0"
      onMouseDown={(event) => event.preventDefault()}
      onClick={() => void copy()}
    >
      {label === "Copied" ? <Check /> : <Copy />}
      {label}
    </Button>
  );
}

function OperatorHints({ ops }: { ops: readonly FilterOp[] }) {
  return (
    <div className="pb-1">
      <div className="px-2 pt-2 pb-1 text-xs text-muted-foreground">Comparison operators</div>
      {OPERATORS.filter((op) => ops.includes(op.op)).map((op) => (
        <div key={op.label} className="flex items-center justify-between px-2 py-1 font-mono text-xs">
          <span className="text-warning">{op.label}</span>
          <code className="rounded bg-muted px-1.5 text-muted-foreground">{op.example}</code>
        </div>
      ))}
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded border border-border px-1 font-sans">
      {children}
    </kbd>
  );
}

export const SearchBox = { Root, Input, Suggestions, CopyCommand } as const;
