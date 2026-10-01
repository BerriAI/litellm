"use client";

import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { FoldChevron } from "./Collapse";

const LONG_VALUE_CHARS = 90;
const ID_KEY = /(^|_)id$/i;

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

const ROW = "flex min-w-0 items-baseline gap-2.5 py-0.5";
const KEY = "shrink-0 text-[13px] leading-[1.2] font-medium tracking-[-0.26px] whitespace-nowrap";

function Lead({ children }: { children: React.ReactNode }) {
  return <span className="flex size-4 shrink-0 items-center justify-center self-center">{children}</span>;
}

function KeyValueRow({ entry, mono }: { entry: KeyValue; mono: boolean }) {
  const [key, value] = entry;
  const [open, setOpen] = useState(false);
  const valueClass = cn(
    "min-w-0 flex-1 text-left text-[13px] leading-[1.2] tracking-[-0.26px] text-trace-text",
    mono || ID_KEY.test(key) ? "font-mono" : "font-sans",
  );

  if (!isLongValue(value)) {
    return (
      <li className={ROW}>
        <Lead>
          <span className="size-2 rounded-full bg-trace-dot" />
        </Lead>
        <span className={cn(KEY, "text-trace-key")}>{key}</span>
        <span className={cn(valueClass, "truncate")} title={value}>
          {value}
        </span>
      </li>
    );
  }

  const head = (
    <>
      <Lead>
        <FoldChevron open={open} className="size-3 text-trace-text" />
      </Lead>
      <span className={cn(KEY, "text-trace-duration")}>{key}</span>
    </>
  );
  const toggle = {
    type: "button",
    "aria-expanded": open,
    "aria-label": `${open ? "Collapse" : "Expand"} ${key}`,
    onClick: () => setOpen((prev) => !prev),
  } as const;

  return (
    <li className={ROW}>
      {open ? (
        <>
          <button {...toggle} className="flex shrink-0 cursor-pointer items-baseline gap-2.5 text-left">
            {head}
          </button>
          <pre className={cn(valueClass, "leading-[1.5] break-words whitespace-pre-wrap")}>{value}</pre>
        </>
      ) : (
        <button {...toggle} className="flex min-w-0 flex-1 cursor-pointer items-baseline gap-2.5 text-left">
          {head}
          <span className={cn(valueClass, "truncate")}>{value.replace(/\n/g, "\\n")}</span>
        </button>
      )}
    </li>
  );
}

/** Dot-bulleted key / value list; long values show a chevron and expand in place on click. */
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
    <ul className={cn("flex flex-col gap-3", className)}>
      {entries.map((entry) => (
        <KeyValueRow key={entry[0]} entry={entry} mono={mono} />
      ))}
    </ul>
  );
}
