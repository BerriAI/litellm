import { useId } from "react";
import styles from "./LensIntroduction.module.css";

const dotColors = [
  "fill-sky-400 dark:fill-sky-300",
  "fill-violet-400 dark:fill-violet-300",
  "fill-amber-400 dark:fill-amber-300",
  "fill-cyan-400 dark:fill-cyan-300",
  "fill-rose-400 dark:fill-rose-300",
];

const gridDots = Array.from({ length: 96 }, (_, index) => {
  const variation = (index * 73 + index * index * 19) % 101;
  return {
    x: 5 + (index % 12) * 10,
    y: 4 + Math.floor(index / 12) * 10,
    color: dotColors[variation % dotColors.length],
    opacity: 0.25 + variation * 0.0075,
  };
});

const organizedDots = Array.from({ length: 30 }, (_, index) => ({
  x: 5 + (index % 6) * 10,
  y: 34 + Math.floor(index / 6) * 10,
}));

const lightWaves = [0, 2.1, 4.2].map((phase) =>
  Array.from({ length: 83 }, (_, point) => {
    const x = point * 20 - 320;
    const y = 54 + Math.sin((x / 320) * Math.PI * 2 + phase) * 22;
    return `${point === 0 ? "M" : "L"} ${x} ${y.toFixed(2)}`;
  }).join(" "),
);

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
          <pattern id={`${id}-swarm`} width="120" height="80" patternUnits="userSpaceOnUse">
            {gridDots.map((dot, index) => (
              <circle key={index} cx={dot.x} cy={dot.y} r="1.8" className={dot.color} opacity={dot.opacity} />
            ))}
          </pattern>
          <pattern id={`${id}-processed`} x="500" width="80" height="108" patternUnits="userSpaceOnUse">
            {organizedDots.map((dot, index) => (
              <circle key={index} cx={dot.x} cy={dot.y} r="1.8" className="fill-indigo-500 dark:fill-indigo-400" />
            ))}
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
          <filter id={`${id}-soft-light`} x="-10%" y="-100%" width="120%" height="300%">
            <feGaussianBlur stdDeviation="6" />
          </filter>
          <mask id={`${id}-illumination`} maskUnits="userSpaceOnUse" x="0" y="0" width="530" height="108">
            <rect width="530" height="108" fill="white" opacity="0.35" />
            <g filter={`url(#${id}-soft-light)`}>
              <g fill="none" stroke="white" strokeWidth="26">
                {lightWaves.map((path, index) => (
                  <path
                    key={index}
                    d={path}
                    opacity={0.9 - index * 0.2}
                    className={styles.swarmLight}
                    style={{ animationDuration: `${20 + index * 5}s`, animationDelay: `${index * -9}s` }}
                  />
                ))}
              </g>
            </g>
          </mask>
          <linearGradient
            id={`${id}-rhythm`}
            x1="0"
            x2="160"
            y1="0"
            y2="0"
            gradientUnits="userSpaceOnUse"
            spreadMethod="repeat"
          >
            <stop stopColor="white" stopOpacity="0.5" />
            <stop offset="0.5" stopColor="white" stopOpacity="0.95" />
            <stop offset="1" stopColor="white" stopOpacity="0.5" />
          </linearGradient>
          <mask id={`${id}-organized-light`} maskUnits="userSpaceOnUse" x="500" y="0" width="500" height="108">
            <rect x="180" width="1140" height="108" fill={`url(#${id}-rhythm)`} className={styles.swarmLight} />
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
          <rect x="500" width="500" height="108" fill={`url(#${id}-processed)`} mask={`url(#${id}-organized-light)`} />
          <g mask={`url(#${id}-swarm-shape)`}>
            <g mask={`url(#${id}-illumination)`}>
              <rect width="530" height="108" fill={`url(#${id}-swarm)`} mask={`url(#${id}-before-gate)`} />
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
