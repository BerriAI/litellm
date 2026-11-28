import { memo, useId } from "react";
import { cn } from "@/lib/cva.config";
import { useTranslation } from "react-i18next";
import styles from "./LensIntroduction.module.css";

const dotColors = ["#27b6e8", "#8b5cf6", "#ec6f93", "#e3a42b", "#2aa889"];

const swarmRows = Array.from({ length: 15 }, (_, row) => ({
  y: 6 + row * 9,
  duration: `${[8.8, 9.2, 9.6, 9][row % 4]}s`,
  delay: `${-row * 0.17}s`,
  dots: Array.from({ length: 80 }, (_, column) => {
    const seed = (column % 24) + row * 31;
    const variation = (seed * 73 + seed * seed * 19) % 101;
    return {
      x: column * 10 - 235,
      color: dotColors[variation % dotColors.length],
      opacity: variation < 15 ? 0 : 0.4 + variation * 0.006,
    };
  }),
}));

const organizedColumns = Array.from({ length: 60 }, (_, column) => column - 10);

export const GatewayFlow = memo(function GatewayFlow() {
  const { t } = useTranslation();
  const id = useId();
  return (
    <div className="mt-5 sm:mt-6">
      <div className="grid grid-cols-3 gap-3 font-mono text-xs">
        <div>
          <p className="font-semibold">{t("lensWorkspace.introduction.agents")}</p>
          <p className="mt-0.5 hidden leading-4 text-muted-foreground sm:block">
            {t("lensWorkspace.introduction.agentSteps")}
          </p>
        </div>
        <div className="text-center">
          <p className="font-semibold text-indigo-700 dark:text-indigo-300">
            {t("lensWorkspace.introduction.gateway")}
          </p>
          <p className="mt-0.5 hidden leading-4 text-muted-foreground sm:block">
            {t("lensWorkspace.introduction.infrastructure")}
          </p>
        </div>
        <div className="text-right">
          <p className="font-semibold">{t("lensWorkspace.frameTitle")}</p>
          <p className="mt-0.5 hidden leading-4 text-muted-foreground sm:block">
            {t("lensWorkspace.introduction.findProblems")}
          </p>
        </div>
      </div>
      <svg
        aria-hidden="true"
        viewBox="0 0 1000 200"
        className={cn(styles.flowGraphic, "mt-2 h-auto w-full text-indigo-700 dark:text-indigo-300")}
        focusable="false"
      >
        <defs>
          <pattern id={`${id}-grid`} width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="5" cy="4" r="0.8" className="fill-muted-foreground/15" />
          </pattern>
          <filter id={`${id}-soft-edge`} x="-10%" y="-30%" width="120%" height="160%">
            <feGaussianBlur stdDeviation="3" />
          </filter>
          <clipPath id={`${id}-after-gate`}>
            <rect x="500" width="500" height="160" />
          </clipPath>
          <linearGradient id={`${id}-incoming`} x1="480" x2="530" y1="0" y2="0" gradientUnits="userSpaceOnUse">
            <stop stopColor="white" />
            <stop offset="1" stopColor="black" />
          </linearGradient>
          <mask id={`${id}-before-gate`}>
            <rect width="1000" height="160" fill={`url(#${id}-incoming)`} />
          </mask>
          <mask id={`${id}-swarm-shape`}>
            <g fill="none" stroke="white" strokeWidth="19" strokeLinecap="round" filter={`url(#${id}-soft-edge)`}>
              {[12, 36, 60, 84, 108, 132].map((y) => (
                <path key={y} d={`M -30 ${y} C 130 ${y - 14} 280 ${y + 16} 370 ${(y + 80) / 2} S 450 80 520 80`} />
              ))}
            </g>
          </mask>
          <mask id={`${id}-organized-shape`}>
            <path
              d="M 490 80 C 620 80 664 72 772 80 S 920 90 1020 80"
              fill="none"
              stroke="white"
              strokeWidth="25"
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
            <rect width="1000" height="200" fill={`url(#${id}-fade)`} />
          </mask>
          <radialGradient id={`${id}-glow`}>
            <stop stopColor="currentColor" stopOpacity="0.18" />
            <stop offset="1" stopColor="currentColor" stopOpacity="0" />
          </radialGradient>
        </defs>
        <g mask={`url(#${id}-edges)`}>
          <rect width="1000" height="200" fill={`url(#${id}-grid)`} />
          <ellipse cx="500" cy="80" rx="44" ry="40" fill={`url(#${id}-glow)`} className={styles.gatewayGlow} />
          <g clipPath={`url(#${id}-after-gate)`}>
            <g mask={`url(#${id}-organized-shape)`}>
              <g className={styles.organizedDots}>
                {organizedColumns.map((column) => (
                  <g key={column} opacity={0.6 + ((column + 10) % 5) * 0.08}>
                    {[62, 71, 80, 89, 98].map((y, row) => (
                      <circle
                        key={y}
                        cx={505 + column * 10}
                        cy={y}
                        r="2"
                        fill={dotColors[(column + 10 + row * 3) % dotColors.length]}
                      />
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
                  style={{
                    animationDuration: row.duration,
                    animationDelay: row.delay,
                  }}
                >
                  {row.dots.map((dot, column) => (
                    <circle key={column} cx={dot.x} cy={row.y} r="2" fill={dot.color} opacity={dot.opacity} />
                  ))}
                </g>
              ))}
            </g>
          </g>
        </g>
        <path
          d="M 965 96 C 965 207 500 207 500 106"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeDasharray="1 7"
          strokeLinecap="round"
          opacity="0.65"
          className={styles.insightLoop}
        />
        <path
          d="m 859 164 -9 7 11 3 M 728 176 l -10 5 10 5 M 577 158 l -11 1 6 9"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
        <rect x="497" y="55" width="6" height="50" rx="3" className="fill-background/90" />
        <path
          d="M 487 55 H 497 V 105 H 487 M 513 55 H 503 V 105 H 513"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
        <circle cx="500" cy="80" r="5" className="fill-background" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="500" cy="80" r="2" fill="currentColor" />
      </svg>
      <p className="ml-auto w-1/2 text-center font-mono text-xs text-indigo-700  dark:text-indigo-300">
        {t("lensWorkspace.introduction.nextRun")}
      </p>
    </div>
  );
});
