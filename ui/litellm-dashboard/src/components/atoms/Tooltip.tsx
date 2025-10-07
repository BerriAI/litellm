import React, { useState, useRef } from "react";
import { QuestionCircleOutlined } from "@ant-design/icons";

interface TooltipProps {
  content: React.ReactNode;
  children?: React.ReactNode;
  width?: string;
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ content, children, width = "auto", className = "" }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState<"top" | "bottom">("top");
  const tooltipRef = useRef<HTMLDivElement>(null);

  // Function to check if tooltip would fit above
  const checkTooltipPosition = () => {
    if (tooltipRef.current) {
      const rect = tooltipRef.current.getBoundingClientRect();
      const tooltipHeight = 300; // Approximate height of the tooltip
      const spaceAbove = rect.top;
      const spaceBelow = window.innerHeight - rect.bottom;

      if (spaceAbove < tooltipHeight && spaceBelow > tooltipHeight) {
        setTooltipPosition("bottom");
      } else {
        setTooltipPosition("top");
      }
    }
  };

  return (
    <div className="relative inline-block" ref={tooltipRef}>
      {children || (
        <QuestionCircleOutlined
          className="ml-1 text-gray-500 cursor-help"
          onMouseEnter={() => {
            checkTooltipPosition();
            setShowTooltip(true);
          }}
          onMouseLeave={() => setShowTooltip(false)}
        />
      )}
      {showTooltip && (
        <div
          className={`absolute left-1/2 -translate-x-1/2 z-50 bg-black/90 text-white p-2 rounded-md text-sm font-normal shadow-lg ${className}`}
          style={{
            [tooltipPosition === "top" ? "bottom" : "top"]: "100%",
            width: width,
            marginBottom: tooltipPosition === "top" ? "8px" : "0",
            marginTop: tooltipPosition === "bottom" ? "8px" : "0",
          }}
        >
          {content}
          <div
            className="absolute left-1/2 -translate-x-1/2 w-0 h-0"
            style={{
              top: tooltipPosition === "top" ? "100%" : "auto",
              bottom: tooltipPosition === "bottom" ? "100%" : "auto",
              borderTop: tooltipPosition === "top" ? "6px solid rgba(0, 0, 0, 0.9)" : "6px solid transparent",
              borderBottom: tooltipPosition === "bottom" ? "6px solid rgba(0, 0, 0, 0.9)" : "6px solid transparent",
              borderLeft: "6px solid transparent",
              borderRight: "6px solid transparent",
            }}
          />
        </div>
      )}
    </div>
  );
};
