import { useId } from "react";
import styles from "./LensIntroduction.module.css";

const dotColors = [
  "fill-sky-400 dark:fill-sky-300",
  "fill-violet-400 dark:fill-violet-300",
  "fill-amber-400 dark:fill-amber-300",
  "fill-cyan-400 dark:fill-cyan-300",
  "fill-rose-400 dark:fill-rose-300",
];

const gridDots = Array.from({ length: 583 }, (_, index) => {
  const column = index % 53;
  const row = Math.floor(index / 53);
  const variation = (index * 73 + index * index * 19) % 101;
  return {
    x: 5 + column * 10,
    y: 4 + row * 10,
    color: dotColors[variation % dotColors.length],
    duration: `${5.5 + (row % 3) * 0.7}s`,
    delay: `${-((53 - column) * 0.1 + row * 0.43 + variation * 0.016)}s`,
  };
});

const organizedColumns = Array.from({ length: 50 }, (_, column) => column).filter((column) => column % 8 < 6);

export function GatewayFlow() {
  const id = useId();
  return (
    <div className="mt-5 sm:mt-6">
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div>
          <p className="font-semibold">Agent swarms</p>
          <p className="mt-0.5 leading-4 text-muted-foreground">Every run, every recorded step</p>
        </div>
        <div className="text-center">
          <p className="font-semibold">LiteLLM gateway</p>
          <p className="mt-0.5 leading-4 text-muted-foreground">One place, your infrastructure</p>
        </div>
        <div className="text-right">
          <p className="font-semibold">Lens</p>
          <p className="mt-0.5 leading-4 text-muted-foreground">Findings to improve your agents</p>
        </div>
      </div>
      <svg
        aria-hidden="true"
        viewBox="0 0 1000 108"
        className={`${styles.flowGraphic} mt-2 h-auto w-full`}
        focusable="false"
      >
        <defs>
          <pattern id={`${id}-grid`} width="10" height="10" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="4" r="1" className="fill-muted-foreground/15" />
          </pattern>
          <linearGradient id={`${id}-incoming`} x1="480" x2="530" y1="0" y2="0" gradientUnits="userSpaceOnUse">
            <stop stopColor="white" />
            <stop offset="1" stopColor="black" />
          </linearGradient>
          <mask id={`${id}-before-gate`}>
            <rect width="1000" height="108" fill={`url(#${id}-incoming)`} />
          </mask>
          <mask id={`${id}-swarm-shape`}>
            <path
              d="M 0 5 C 200 5 270 8 380 30 S 465 39 500 39 C 620 39 730 18 1000 12 L 1000 96 C 730 90 620 69 500 69 C 465 69 440 69 380 78 S 200 103 0 103 Z"
              fill="white"
            />
          </mask>
          <linearGradient id={`${id}-fade`}>
            <stop offset="0" stopColor="white" stopOpacity="0" />
            <stop offset="0.06" stopColor="white" />
            <stop offset="0.94" stopColor="white" />
            <stop offset="1" stopColor="white" stopOpacity="0" />
          </linearGradient>
          <mask id={`${id}-edges`}>
            <rect width="1000" height="108" fill={`url(#${id}-fade)`} />
          </mask>
          <radialGradient id={`${id}-glow`} className="text-indigo-500 dark:text-indigo-400">
            <stop stopColor="currentColor" stopOpacity="0.2" />
            <stop offset="1" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
          <linearGradient id={`${id}-gate`} x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="currentColor" className="text-sky-400" />
            <stop offset="0.5" stopColor="currentColor" className="text-indigo-500 dark:text-indigo-300" />
            <stop offset="1" stopColor="currentColor" className="text-violet-400" />
          </linearGradient>
        </defs>
        <g mask={`url(#${id}-edges)`}>
          <rect width="1000" height="108" fill={`url(#${id}-grid)`} />
          <ellipse cx="500" cy="54" rx="58" ry="54" fill={`url(#${id}-glow)`} className={styles.gatewayGlow} />
          {organizedColumns.map((column) => (
            <g
              key={column}
              className={`${styles.organizedDots} fill-indigo-500 dark:fill-indigo-400`}
              style={{ animationDelay: `${-(50 - column) * 0.1}s` }}
            >
              {[34, 44, 54, 64, 74].map((y) => (
                <circle key={y} cx={505 + column * 10} cy={y} r="1.8" />
              ))}
            </g>
          ))}
          <g mask={`url(#${id}-swarm-shape)`}>
            <g mask={`url(#${id}-before-gate)`}>
              {gridDots.map((dot, index) => (
                <circle
                  key={index}
                  cx={dot.x}
                  cy={dot.y}
                  r="1.8"
                  className={`${styles.swarmDot} ${dot.color}`}
                  style={{ animationDuration: dot.duration, animationDelay: dot.delay }}
                />
              ))}
            </g>
          </g>
        </g>
        <rect x="497" y="10" width="6" height="88" rx="3" className="fill-background/90" />
        <path
          d="M 490 10 H 497 V 98 H 490 M 510 10 H 503 V 98 H 510"
          fill="none"
          stroke={`url(#${id}-gate)`}
          strokeWidth="1.4"
          opacity="0.75"
        />
        <circle cx="500" cy="54" r="4" className="fill-background stroke-indigo-400" strokeWidth="1.4" />
        <circle cx="500" cy="54" r="1.6" className="fill-indigo-500 dark:fill-indigo-300" />
      </svg>
    </div>
  );
}
