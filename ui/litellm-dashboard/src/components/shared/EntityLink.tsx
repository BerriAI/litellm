"use client";

import { ChevronRight } from "lucide-react";
import { useRouter } from "next/navigation";
import * as React from "react";

import { cn } from "@/lib/cva.config";

export function useEntityLinkClick(href: string): (e: React.MouseEvent) => void {
  const router = useRouter();

  return (e: React.MouseEvent) => {
    const hasModifierKey = e.metaKey || e.ctrlKey || e.shiftKey;
    const isNativeNewTabClick = hasModifierKey || e.button === 1;
    if (isNativeNewTabClick) return;
    e.preventDefault();
    router.push(href);
  };
}

interface EntityLinkProps {
  href: string;
  className?: string;
  children: React.ReactNode;
}

export function EntityLink({ href, className, children }: EntityLinkProps) {
  const handleClick = useEntityLinkClick(href);

  return (
    <a
      href={href}
      onClick={handleClick}
      className={cn(
        "group inline-flex min-w-0 max-w-full items-center gap-0.5 font-semibold underline-offset-4 hover:underline",
        className,
      )}
    >
      <span className="min-w-0 truncate">{children}</span>
      <ChevronRight className="size-3.5 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground" />
    </a>
  );
}
