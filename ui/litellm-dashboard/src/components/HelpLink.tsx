import React, { useState, useRef, useEffect } from "react";
import { ExternalLink, ChevronDown } from "lucide-react";

interface HelpLinkProps {
  href: string;
  children?: React.ReactNode;
  variant?: "inline" | "subtle" | "button";
  className?: string;
}

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
 * A reusable component for linking to documentation, styled similar to Linear's help links.
 * 
 * @example
 * // Inline "Learn more" style
 * <HelpLink href="https://docs.litellm.ai/docs/proxy/custom_pricing">
 *   Learn more about custom pricing
 * </HelpLink>
 * 
 * @example
 * // Subtle link (just icon + text, minimal styling)
 * <HelpLink href="https://docs.litellm.ai/docs/proxy/cost_tracking" variant="subtle">
 *   View docs
 * </HelpLink>
 * 
 * @example
 * // Button style (more prominent)
 * <HelpLink href="https://docs.litellm.ai/docs/proxy/custom_pricing" variant="button">
 *   Custom Pricing Documentation
 * </HelpLink>
 */
export const HelpLink: React.FC<HelpLinkProps> = ({
  href,
  children = "Learn more",
  variant = "inline",
  className = "",
}) => {
  const baseClasses = "inline-flex items-center gap-1.5 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 rounded";
  
  const variantClasses = {
    inline: "text-blue-600 hover:text-blue-800 text-sm font-medium hover:underline",
    subtle: "text-gray-500 hover:text-gray-700 text-xs",
    button: "text-blue-600 hover:text-blue-700 border border-gray-200 hover:border-gray-300 px-3 py-1.5 rounded-md bg-white hover:bg-gray-50 text-sm font-medium shadow-sm",
  };

  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      title="Open documentation in a new tab"
    >
      <span>{children}</span>
      <ExternalLink className="h-3.5 w-3.5 flex-shrink-0" aria-hidden="true" />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  );
};

/**
 * A minimal help icon with tooltip for inline contextual help.
 * Similar to Linear's "?" icons that appear next to labels.
 */
interface HelpIconProps {
  content: React.ReactNode;
  learnMoreHref?: string;
  learnMoreText?: string;
}

export const HelpIcon: React.FC<HelpIconProps> = ({
  content,
  learnMoreHref,
  learnMoreText = "Learn more",
}) => {
  const [showTooltip, setShowTooltip] = React.useState(false);

  return (
    <div className="relative inline-block ml-1.5">
      <button
        type="button"
        className="inline-flex items-center justify-center w-4 h-4 text-gray-400 hover:text-gray-600 transition-colors cursor-help focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-full"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        onFocus={() => setShowTooltip(true)}
        onBlur={() => setShowTooltip(false)}
        aria-label="Help information"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="10" strokeWidth="1.5" />
          <path strokeLinecap="round" d="M12 17h0M12 13.5a1.5 1.5 0 0 1 1-1.415A1.5 1.5 0 1 0 12 9" strokeWidth="1.5" />
        </svg>
      </button>
      {showTooltip && (
        <div
          className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-50 bg-gray-900 text-white p-3 rounded-lg text-xs shadow-lg w-64"
          style={{ pointerEvents: "none" }}
        >
          <div className="mb-2">{content}</div>
          {learnMoreHref && (
            <a
              href={learnMoreHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-blue-300 hover:text-blue-200 font-medium"
              style={{ pointerEvents: "auto" }}
            >
              {learnMoreText}
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </a>
          )}
          <div
            className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0"
            style={{
              borderTop: "6px solid rgb(17 24 39)",
              borderLeft: "6px solid transparent",
              borderRight: "6px solid transparent",
            }}
          />
        </div>
      )}
    </div>
  );
};

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
export const DocsMenu: React.FC<DocsMenuProps> = ({
  items,
  children = "Docs",
  className = "",
}) => {
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
        className="inline-flex items-center gap-1 text-gray-500 hover:text-gray-700 text-xs transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 rounded px-2 py-1"
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <span>{children}</span>
        <ChevronDown className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-1 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50">
          {items.map((item, index) => (
            <a
              key={index}
              href={item.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-between px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
              onClick={() => setIsOpen(false)}
            >
              <span>{item.label}</span>
              <ExternalLink className="h-3.5 w-3.5 text-gray-400 flex-shrink-0 ml-2" aria-hidden="true" />
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

