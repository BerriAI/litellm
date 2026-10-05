import type { ComponentProps, ReactNode } from "react";
import { cva, cn } from "@/lib/cva.config";

const stateIcon = cva("flex size-10 items-center justify-center rounded-full", {
  variants: {
    tone: { muted: "bg-muted text-muted-foreground", destructive: "bg-destructive/10 text-destructive" },
  },
  defaultVariants: { tone: "muted" },
});

export type StateMessageProps = Omit<ComponentProps<"div">, "title"> & {
  role: "status" | "alert";
  icon: ReactNode;
  tone?: "muted" | "destructive";
  title: string;
  description: ReactNode;
};

export function StateMessage({ icon, tone, title, description, children, className, ...props }: StateMessageProps) {
  return (
    <div
      data-slot="state-message"
      className={cn(
        "m-auto flex max-w-sm flex-col items-center gap-3 py-16 text-center animate-in fade-in-0 duration-300 motion-reduce:animate-none",
        className,
      )}
      {...props}
    >
      <span aria-hidden="true" className={stateIcon({ tone })}>
        {icon}
      </span>
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {children && <div className="mt-1 flex items-center gap-2">{children}</div>}
    </div>
  );
}
