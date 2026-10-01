"use client";

import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { Collapse, FoldChevron } from "./Collapse";

const LONG_VALUE_CHARS = 90;

export type KeyValue = readonly [key: string, value: string];

export const isLongValue = (value: string): boolean => value.length > LONG_VALUE_CHARS || value.includes("\n");

export const displayValue = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return String(value);
  return JSON.stringify(value);
};

export const objectEntries = (value: unknown): KeyValue[] | null => {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  return Object.entries(value).map(([key, v]): KeyValue => [key, displayValue(v)]);
};

function KeyValueRow({ entry, mono }: { entry: KeyValue; mono: boolean }) {
  const [key, value] = entry;
  const long = isLongValue(value);
  const [open, setOpen] = useState(false);
  const marker = long ? (
    <FoldChevron open={open} className="size-3" />
  ) : (
    <i className="size-1.5 rounded-full bg-border" />
  );
  const head = (
    <>
      <span className="grid h-5 w-3 shrink-0 place-items-center text-muted-foreground">{marker}</span>
      <span className="shrink-0 font-medium text-trace-key">{key}</span>
      <span
        className={cn(
          "min-w-0 flex-1 truncate text-left text-trace-text transition-opacity duration-150",
          mono && "font-mono",
          open && "opacity-40",
        )}
      >
        {value.replace(/\n/g, "\\n")}
      </span>
    </>
  );
  if (!long) return <li className="flex min-w-0 items-start gap-2.5 py-1">{head}</li>;
  return (
    <li>
      <button
        type="button"
        aria-expanded={open}
        aria-label={`${open ? "Collapse" : "Expand"} ${key}`}
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full min-w-0 items-start gap-2.5 py-1 text-left"
      >
        {head}
      </button>
      <Collapse open={open}>
        <pre className="mt-0.5 mb-1.5 ml-[22px] rounded-md bg-muted px-2.5 py-2 font-mono text-[12px] break-words whitespace-pre-wrap text-foreground">
          {value}
        </pre>
      </Collapse>
    </li>
  );
}

/** Dot-bulleted key / value list; long values collapse to one line and expand on click. */
export function KeyValueRows({
  entries,
  mono = false,
  className,
}: {
  entries: readonly KeyValue[];
  mono?: boolean;
  className?: string;
}) {
  return (
    <ul className={cn("flex flex-col text-[13px]", className)}>
      {entries.map((entry) => (
        <KeyValueRow key={entry[0]} entry={entry} mono={mono} />
      ))}
    </ul>
  );
}
