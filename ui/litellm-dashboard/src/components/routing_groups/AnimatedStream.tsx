"use client";
import React from "react";

interface AnimatedStreamProps {
  /** SVG path d string */
  path: string;
  color: string;
  /** stroke-width in px */
  thickness: number;
  animated?: boolean;
}

export default function AnimatedStream({ path, color, thickness, animated = true }: AnimatedStreamProps) {
  const dashLen = Math.max(8, thickness * 3);
  const gapLen = Math.max(4, thickness * 2);

  return (
    <g>
      {/* Dim base */}
      <path d={path} fill="none" stroke={color} strokeWidth={thickness} strokeOpacity={0.12} />
      {/* Glow */}
      <path d={path} fill="none" stroke={color} strokeWidth={thickness + 6} strokeOpacity={0.06} />
      {/* Animated dashes */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={thickness}
        strokeOpacity={0.7}
        strokeDasharray={`${dashLen} ${gapLen}`}
        style={
          animated
            ? {
                animation: `routingFlow 0.9s linear infinite`,
                strokeDashoffset: 0,
              }
            : undefined
        }
      />
      <style>{`
        @keyframes routingFlow {
          from { stroke-dashoffset: ${dashLen + gapLen}; }
          to   { stroke-dashoffset: 0; }
        }
      `}</style>
    </g>
  );
}
