import { useId } from "react";
import styles from "./LensIntroduction.module.css";

const streams = [
  {
    id: "runs",
    color: "text-sky-500 dark:text-sky-400",
    path: "M -32 66 C 24 66 32 22 96 22 S 192 86 256 86 S 368 28 422 28 C 464 28 472 54 500 54 C 558 54 584 86 646 86 S 756 22 818 22 S 924 86 986 86 S 1040 54 1056 54",
  },
  {
    id: "steps",
    color: "text-violet-500 dark:text-violet-400",
    path: "M -32 30 C 40 30 58 78 120 78 S 218 24 284 24 S 390 76 444 76 C 470 76 480 54 503 54",
  },
  {
    id: "tools",
    color: "text-amber-500 dark:text-amber-400",
    path: "M -32 54 C 26 54 68 68 128 68 S 230 38 288 38 S 396 62 448 62 C 474 62 484 54 503 54",
  },
];

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
          <pattern id={`${id}-field`} width="12" height="12" patternUnits="userSpaceOnUse">
            <circle cx="6" cy="6" r="1.15" className="fill-muted-foreground/15" />
          </pattern>
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
          {streams.map((stream) => (
            <linearGradient
              key={stream.id}
              id={`${id}-${stream.id}`}
              x1="470"
              x2="530"
              y1="0"
              y2="0"
              gradientUnits="userSpaceOnUse"
              className={stream.color}
            >
              <stop stopColor="currentColor" />
              <stop offset="1" stopColor="currentColor" className="text-indigo-500 dark:text-indigo-400" />
            </linearGradient>
          ))}
        </defs>
        <g mask={`url(#${id}-edges)`}>
          <rect width="1000" height="108" fill={`url(#${id}-field)`} />
          <ellipse cx="500" cy="54" rx="58" ry="54" fill={`url(#${id}-glow)`} className={styles.gatewayGlow} />
          {streams.map((stream) => (
            <g key={stream.id} fill="none" stroke={`url(#${id}-${stream.id})`} strokeLinecap="round">
              <path d={stream.path} strokeWidth="1" opacity="0.13" />
              <path d={stream.path} strokeWidth="2.4" className={styles.trails} />
              <path d={stream.path} strokeWidth="3.6" className={styles.packets} />
            </g>
          ))}
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
