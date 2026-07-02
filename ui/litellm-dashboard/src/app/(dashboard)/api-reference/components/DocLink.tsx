import React from "react";
import { ExternalLink } from "lucide-react";

import { cn } from "@/lib/cva.config";

export type DocLinkProps = {
  href?: string;
  className?: string;
};

const DocLink = ({ href, className }: DocLinkProps) => {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      title="Open documentation in a new tab"
      className={cn(
        "inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white/80 px-3.5 py-2 text-sm font-medium text-zinc-700 shadow-xs",
        "hover:bg-white focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-blue-500 active:translate-y-[0.5px]",
        className,
      )}
    >
      <span>API Reference Docs</span>
      <ExternalLink aria-hidden className="h-4 w-4 opacity-80" />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
};

export default DocLink;
