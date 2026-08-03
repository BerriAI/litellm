"use client";

import { useRouter } from "next/navigation";
import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

const ENTITY_BADGE_SIZE = "px-2.5 py-1 text-sm";

interface BadgeLinkProps {
  href?: string;
  variant?: React.ComponentProps<typeof Badge>["variant"];
  className?: string;
  children: React.ReactNode;
}

export function BadgeLink({ href, variant = "secondary", className, children }: BadgeLinkProps) {
  const router = useRouter();

  if (!href) {
    return (
      <Badge variant={variant} className={cn(ENTITY_BADGE_SIZE, className)}>
        {children}
      </Badge>
    );
  }

  const handleClick = (e: React.MouseEvent) => {
    const hasModifierKey = e.metaKey || e.ctrlKey || e.shiftKey;
    const isNativeNewTabClick = hasModifierKey || e.button === 1;
    if (isNativeNewTabClick) return;
    e.preventDefault();
    router.push(href);
  };

  return (
    <Badge
      variant={variant}
      className={cn("cursor-pointer", ENTITY_BADGE_SIZE, className)}
      render={<a href={href} onClick={handleClick} />}
    >
      {children}
    </Badge>
  );
}
