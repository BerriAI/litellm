import { CircleHelp } from "lucide-react";
import React, { type ReactNode } from "react";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import type { MathRow } from "./usageUnits";

export function CalcPopover({ title, formula, children }: { title: string; formula: string; children: ReactNode }) {
  return (
    <Popover>
      <PopoverTrigger
        openOnHover
        delay={200}
        closeDelay={150}
        render={
          <button
            type="button"
            className="mt-2 inline-flex w-fit cursor-help items-start gap-1 text-left text-xs text-muted-foreground hover:text-foreground"
          />
        }
      >
        <CircleHelp className="mt-px size-3.5 shrink-0" />
        How is this calculated?
      </PopoverTrigger>
      <PopoverContent side="bottom" align="start" className="w-auto min-w-72 max-w-md gap-3">
        <PopoverTitle>{title}</PopoverTitle>
        <code className="w-fit rounded bg-muted px-2 py-1 text-[11px] text-muted-foreground">{formula}</code>
        {children}
      </PopoverContent>
    </Popover>
  );
}

export function MathTable({ rows, total }: { rows: readonly MathRow[]; total: string }) {
  const width = 1 + Math.max(...rows.map((row) => row.parts.length), 1);
  return (
    <table className="w-full text-xs">
      <tbody>
        {rows.map((row) => (
          <React.Fragment key={row.label}>
            <tr>
              <td className="py-0.5 pr-3">{row.label}</td>
              {row.parts.map((part, i) => (
                <td key={i} className="py-0.5 pl-3 text-right whitespace-nowrap tabular-nums">
                  {part}
                </td>
              ))}
            </tr>
            {row.note && (
              <tr>
                <td colSpan={width} className="pb-1 text-[11px] text-warning">
                  {row.note}
                </td>
              </tr>
            )}
          </React.Fragment>
        ))}
      </tbody>
      <tfoot>
        <tr className="border-t border-border font-medium">
          <td className="pt-1.5 pr-3" colSpan={width - 1}>
            Total
          </td>
          <td className="pt-1.5 pl-3 text-right whitespace-nowrap tabular-nums">{total}</td>
        </tr>
      </tfoot>
    </table>
  );
}
