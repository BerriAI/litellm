import React from "react";
import { ExternalLink } from "lucide-react";

function cn(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

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
        "inline-flex items-center gap-2 rounded-xl border border-zinc-200 bg-white/80 px-3.5 py-2 text-sm font-medium text-zinc-700 shadow-sm",
        "hover:bg-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 active:translate-y-[0.5px]",
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
