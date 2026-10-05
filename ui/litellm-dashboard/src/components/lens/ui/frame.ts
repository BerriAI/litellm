import { cva } from "@/lib/cva.config";

export const frameCard = cva(
  "flex min-h-0 flex-1 flex-col overflow-hidden rounded-b-2xl bg-card @min-[24rem]/lens-frame:rounded-t-2xl",
  {
    variants: {
      session: {
        live: "border border-t-0 border-foreground/15 @min-[24rem]/lens-frame:border-t",
        demo: "border-2 border-t-0 border-info @min-[24rem]/lens-frame:border-t-2",
      },
    },
    defaultVariants: { session: "live" },
  },
);

export const frameTab = cva("relative z-raised rounded-t-2xl bg-card", {
  variants: {
    session: {
      live: "border-x border-t border-foreground/15 @min-[24rem]/lens-frame:-mb-px",
      demo: "border-x-2 border-t-2 border-info @min-[24rem]/lens-frame:-mb-0.5",
    },
  },
  defaultVariants: { session: "live" },
});

export const frameCorner = cva("block size-full shadow-[0_0_0_12px_var(--card)]", {
  variants: {
    session: { live: "border-foreground/15", demo: "border-info" },
    side: { left: "rounded-br-xl", right: "rounded-bl-xl" },
  },
  compoundVariants: [
    { session: "live", side: "left", className: "border-r border-b" },
    { session: "live", side: "right", className: "border-l border-b" },
    { session: "demo", side: "left", className: "border-r-2 border-b-2" },
    { session: "demo", side: "right", className: "border-l-2 border-b-2" },
  ],
  defaultVariants: { session: "live" },
});
