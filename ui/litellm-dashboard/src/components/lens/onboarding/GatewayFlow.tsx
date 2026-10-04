import { useId } from "react";
import { cn } from "@/lib/cva.config";
import styles from "./LensIntroduction.module.css";

const dotColors = [
  "fill-sky-400 dark:fill-sky-300",
  "fill-indigo-400 dark:fill-indigo-300",
  "fill-violet-400 dark:fill-violet-300",
];

const swarmRows = Array.from({ length: 11 }, (_, row) => ({
  y: 4 + row * 10,
  duration: `${[4.8, 5.2, 5.6, 5][row % 4]}s`,
  delay: `${-row * 0.17}s`,
  dots: Array.from({ length: 80 }, (_, column) => {
    const seed = (column % 24) + row * 31;
    const variation = (seed * 73 + seed * seed * 19) % 101;
    return {
      x: column * 10 - 235,
      color: dotColors[variation % dotColors.length],
      opacity: variation < 15 ? 0 : 0.5 + variation * 0.005,
    };
  }),
}));

const organizedColumns = Array.from({ length: 58 }, (_, column) => column - 8);

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
        className={cn(styles.flowGraphic, "mt-2 h-auto w-full")}
        focusable="false"
      >
        <defs>
          <pattern id={`${id}-grid`} width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="4" r="1" className="fill-muted-foreground/15" />
          </pattern>
          <filter id={`${id}-soft-edge`} x="-10%" y="-30%" width="120%" height="160%">
            <feGaussianBlur stdDeviation="3" />
          </filter>
          <clipPath id={`${id}-after-gate`}>
            <rect x="500" width="500" height="108" />
          </clipPath>
          <linearGradient id={`${id}-incoming`} x1="480" x2="530" y1="0" y2="0" gradientUnits="userSpaceOnUse">
            <stop stopColor="white" />
            <stop offset="1" stopColor="black" />
          </linearGradient>
          <mask id={`${id}-before-gate`}>
            <rect width="1000" height="108" fill={`url(#${id}-incoming)`} />
          </mask>
          <mask id={`${id}-swarm-shape`}>
            <g fill="none" stroke="white" strokeWidth="24" strokeLinecap="round" filter={`url(#${id}-soft-edge)`}>
              <path d="M -30 20 C 80 0 142 38 244 22 S 402 36 530 54" />
              <path d="M -30 54 C 90 76 174 32 278 52 S 420 60 530 54" />
              <path d="M -30 90 C 80 112 172 66 276 86 S 420 68 530 54" />
            </g>
          </mask>
          <mask id={`${id}-organized-shape`}>
            <path
              d="M 490 54 C 620 54 664 66 772 58 S 920 42 1020 52"
              fill="none"
              stroke="white"
              strokeWidth="32"
              filter={`url(#${id}-soft-edge)`}
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
            <stop stopColor="currentColor" stopOpacity="0.14" />
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
          <ellipse cx="500" cy="54" rx="44" ry="40" fill={`url(#${id}-glow)`} className={styles.gatewayGlow} />
          <g clipPath={`url(#${id}-after-gate)`}>
            <g mask={`url(#${id}-organized-shape)`}>
              <g className={cn(styles.organizedDots, "fill-indigo-500 dark:fill-indigo-400")}>
                {organizedColumns.map((column) => (
                  <g key={column} opacity={0.6 + (((column + 8) * 3) % 8) * 0.05}>
                    {[34, 44, 54, 64, 74].map((y) => (
                      <circle key={y} cx={505 + column * 10} cy={y} r="2.3" />
                    ))}
                  </g>
                ))}
              </g>
            </g>
          </g>
          <g mask={`url(#${id}-swarm-shape)`}>
            <g mask={`url(#${id}-before-gate)`}>
              {swarmRows.map((row, index) => (
                <g
                  key={index}
                  className={styles.swarmRow}
                  style={{ animationDuration: row.duration, animationDelay: row.delay }}
                >
                  {row.dots.map((dot, column) => (
                    <circle key={column} cx={dot.x} cy={row.y} r="2.3" className={dot.color} opacity={dot.opacity} />
                  ))}
                </g>
              ))}
            </g>
          </g>
        </g>
        <rect x="497" y="30" width="6" height="48" rx="3" className="fill-background/90" />
        <path
          d="M 490 30 H 497 V 78 H 490 M 510 30 H 503 V 78 H 510"
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
