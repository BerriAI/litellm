import React, { useState, useRef, useEffect } from "react";
import { ExternalLink, ChevronDown } from "lucide-react";

interface DocMenuItem {
  label: string;
  href: string;
}

interface DocsMenuProps {
  items: DocMenuItem[];
  children?: React.ReactNode;
  className?: string;
}

/**
 * A dropdown menu for multiple documentation links.
 * Linear-style: Single "Docs" button that expands to show multiple relevant links.
 *
 * @example
 * <DocsMenu items={[
 *   { label: "Custom pricing for models", href: "https://docs.litellm.ai/docs/proxy/custom_pricing" },
 *   { label: "Spend tracking", href: "https://docs.litellm.ai/docs/proxy/cost_tracking" }
 * ]}>
 *   Docs
 * </DocsMenu>
 */
export const DocsMenu: React.FC<DocsMenuProps> = ({ items, children = "Docs", className = "" }) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className={`relative inline-block ${className}`} ref={menuRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground text-xs transition-colors focus:outline-hidden focus:ring-2 focus:ring-ring focus:ring-offset-1 rounded-sm px-2 py-1"
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <span>{children}</span>
        <ChevronDown className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-56 bg-card rounded-lg shadow-lg border border-border py-1 z-floating">
          {items.map((item, index) => (
            <a
              key={index}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-between px-4 py-2 text-sm text-foreground hover:bg-accent transition-colors"
              onClick={() => setIsOpen(false)}
            >
              <span>{item.label}</span>
              <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0 ml-2" aria-hidden="true" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
};
