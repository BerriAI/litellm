"use client";

import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/cva.config";

interface ToolbarSeparatorProps {
  className?: string;
}

export function ToolbarSeparator({ className }: ToolbarSeparatorProps) {
  return <Separator orientation="vertical" className={cn("mx-1.5 h-5 data-vertical:self-center", className)} />;
}
