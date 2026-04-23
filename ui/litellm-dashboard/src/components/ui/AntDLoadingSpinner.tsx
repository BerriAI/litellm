import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface AntDLoadingSpinnerProps {
  size?: "small" | "default" | "large";
  fontSize?: number;
}

/**
 * Back-compat spinner. The name is preserved for a growing number of
 * call sites; the implementation is now a lucide Loader2 sized via
 * Tailwind.
 */
export function AntDLoadingSpinner({ size, fontSize }: AntDLoadingSpinnerProps) {
  const sizeClass = cn(
    size === "small" && "h-3 w-3",
    size === "large" && "h-6 w-6",
    !size || size === "default" ? "h-4 w-4" : "",
  );
  return (
    <Loader2
      className={cn("animate-spin text-primary", sizeClass)}
      style={fontSize ? { fontSize } : undefined}
    />
  );
}
