import { Sparkle } from "lucide-react";
import { cn } from "@/lib/cva.config";

export function AriaAvatar({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <Sparkle
      role="presentation"
      aria-hidden="true"
      className={cn("shrink-0", className)}
      style={{ width: size, height: size }}
      strokeWidth={1.75}
    />
  );
}
