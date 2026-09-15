"use client";

import * as React from "react";

import { useEntityLinkClick } from "@/components/shared/EntityLink";
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
  if (!href) {
    return (
      <Badge variant={variant} className={cn(ENTITY_BADGE_SIZE, className)}>
        {children}
      </Badge>
    );
  }

  return (
    <LinkedBadge href={href} variant={variant} className={className}>
      {children}
    </LinkedBadge>
  );
}

function LinkedBadge({ href, variant, className, children }: BadgeLinkProps & { href: string }) {
  const handleClick = useEntityLinkClick(href);

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
