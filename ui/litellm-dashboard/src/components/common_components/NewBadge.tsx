import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

/**
 * "New" pill or dot indicator. When `dot` is true the indicator is a small
 * absolute-positioned dot in the corner of the wrapped child. When `dot` is
 * false (default) it shows a "New" badge \u2014 inline if no children are
 * passed, otherwise overlapping the child.
 */
export default function NewBadge({
  children,
  dot = false,
}: {
  children?: React.ReactNode;
  dot?: boolean;
}) {
  const disableShowNewBadge = useDisableShowNewBadge();

  if (disableShowNewBadge) {
    return children ? <>{children}</> : null;
  }

  // Standalone (no children): just render the pill / dot.
  if (!children) {
    if (dot) {
      return (
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary align-middle" />
      );
    }
    return (
      <Badge variant="default" className="text-[10px] px-1.5 py-0 h-4 leading-4">
        New
      </Badge>
    );
  }

  // With children: overlap the indicator on the child.
  return (
    <span className="relative inline-block">
      {children}
      <span
        className={cn(
          "absolute -top-0.5 -right-0.5 z-10",
          dot
            ? "block w-1.5 h-1.5 rounded-full bg-primary"
            : "inline-flex items-center rounded-full bg-primary text-primary-foreground text-[9px] leading-none px-1 py-0.5",
        )}
      >
        {!dot && "New"}
      </span>
    </span>
  );
}
