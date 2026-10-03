import type { ComponentProps } from "react";
import { cn } from "@/lib/cva.config";

export type ListRowProps = ComponentProps<"button">;

export function ListRow({ className, ...props }: ListRowProps) {
  return (
    <button
      {...props}
      data-slot="list-row"
      className={cn(
        "w-full py-4 text-left hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-ring",
        className,
      )}
    />
  );
}
