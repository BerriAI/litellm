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

const ROW = "flex min-w-0 items-baseline gap-3 border-b border-border/60 py-2 last:border-0";
const KEY = "w-24 shrink-0 break-words text-xs leading-5 font-normal";

function Lead({ children }: { children: React.ReactNode }) {
  return <span className="flex size-3 shrink-0 items-center justify-center self-center">{children}</span>;
}

function KeyValueRow({ entry, mono }: { entry: KeyValue; mono: boolean }) {
  const [key, value] = entry;
  const [open, setOpen] = useState(false);
  const valueClass = cn(
    "min-w-0 flex-1 break-words text-left text-[13px] leading-5 text-foreground",
    mono || ID_KEY.test(key) ? "font-mono" : "font-sans",
  );

  if (!isLongValue(value)) {
    return (
      <li className={ROW}>
        <span className={cn(KEY, "text-muted-foreground")}>{key}</span>
        <span className={cn(valueClass, "whitespace-pre-wrap")} title={value}>
          {value}
        </span>
      </li>
    );
  }

  const head = (
    <>
      <Lead>
        <FoldChevron open={open} className="size-3 text-foreground" />
      </Lead>
      <span className={cn(KEY, "text-muted-foreground")}>{key}</span>
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
    <ul className={cn("flex flex-col", className)}>
      {entries.map((entry) => (
        <KeyValueRow key={entry[0]} entry={entry} mono={mono} />
      ))}
    </ul>
  );
}
