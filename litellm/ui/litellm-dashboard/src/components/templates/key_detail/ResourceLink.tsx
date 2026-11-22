"use client";

import type { ReactNode } from "react";

import { useEntityLinkClick } from "@/components/shared/EntityLink";
import { cn } from "@/lib/cva.config";

interface ResourceLinkProps {
  href?: string;
  children: ReactNode;
  className?: string;
}

export function ResourceLink({ href, children, className }: ResourceLinkProps) {
  if (!href) return <span className={cn("min-w-0 truncate font-medium", className)}>{children}</span>;
  return (
    <LinkedResource href={href} className={className}>
      {children}
    </LinkedResource>
  );
}

function LinkedResource({ href, children, className }: ResourceLinkProps & { href: string }) {
  const handleClick = useEntityLinkClick(href);
  return (
    <a
      href={href}
      onClick={handleClick}
      className={cn("min-w-0 truncate font-medium text-foreground underline-offset-4 hover:underline", className)}
    >
      {children}
    </a>
  );
}
