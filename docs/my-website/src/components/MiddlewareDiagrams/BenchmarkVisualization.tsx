import React, { useState, useEffect, useRef, useCallback } from 'react';
import styles from './styles.module.css';

/* ── Constants ── */
const TOTAL_REQUESTS = 50_000;
const DURATION_AFTER_MS = 8_000; // "After" column finishes in 8s
const DURATION_BEFORE_MS = 13_920; // 74% slower → 8000 * 1.74
const TICK_MS = 50;
const RESET_PAUSE_MS = 2_000;
const MAX_DOTS = 14;

const BEFORE_RPS = 3_785;
const AFTER_RPS = 6_577;
const BEFORE_P50 = 21;
const AFTER_P50 = 13;

const BEFORE_LAYERS = [
  { label: 'ab client', warning: false },
  { label: 'uvicorn \u00B7 1 worker', warning: false },
  { label: 'ASGI Middleware', warning: false },
  { label: 'BaseHTTPMiddleware', warning: true },
  { label: 'GET /health \u2192 "ok"', warning: false },
];

const AFTER_LAYERS = [
  { label: 'ab client', warning: false },
  { label: 'uvicorn \u00B7 1 worker', warning: false },
  { label: 'ASGI Middleware', warning: false },
  { label: 'ASGI Middleware', warning: false },
  { label: 'GET /health \u2192 "ok"', warning: false },
];

const BENCHMARK_RUNS = [
  { config: 'Before (1 ASGI + 1 BaseHTTP)', run: 1, rps: 3596, p50: 21 },
  { config: 'Before (1 ASGI + 1 BaseHTTP)', run: 2, rps: 3599, p50: 21 },
  { config: 'Before (1 ASGI + 1 BaseHTTP)', run: 3, rps: 4161, p50: 21 },
  { config: 'After (2x Pure ASGI)', run: 1, rps: 6504, p50: 13 },
  { config: 'After (2x Pure ASGI)', run: 2, rps: 6631, p50: 13 },
  { config: 'After (2x Pure ASGI)', run: 3, rps: 6595, p50: 13 },
];

/* ── Dot type ── */
interface Dot {
  id: number;
  progress: number; // 0..1 (top to bottom)
}

/* ── Component ── */
export default function BenchmarkVisualization() {
  const [elapsed, setElapsed] = useState(0);
  const [running, setRunning] = useState(false);
  const [afterDone, setAfterDone] = useState(false);
  const [beforeDone, setBeforeDone] = useState(false);
  const [tableOpen, setTableOpen] = useState(false);
  const [beforeDots, setBeforeDots] = useState<Dot[]>([]);
  const [afterDots, setAfterDots] = useState<Dot[]>([]);
  const dotIdRef = useRef(0);
  const observerRef = useRef<IntersectionObserver | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const hasStartedRef = useRef(false);

  const beforeProgress = Math.min(elapsed / DURATION_BEFORE_MS, 1);
  const afterProgress = Math.min(elapsed / DURATION_AFTER_MS, 1);
  const beforeCompleted = Math.round(beforeProgress * TOTAL_REQUESTS);
  const afterCompleted = Math.round(afterProgress * TOTAL_REQUESTS);
  const beforeCurrentRPS = running && !beforeDone
    ? Math.round(BEFORE_RPS * (0.9 + Math.random() * 0.2))
    : beforeDone ? 0 : 0;
  const afterCurrentRPS = running && !afterDone
    ? Math.round(AFTER_RPS * (0.9 + Math.random() * 0.2))
    : afterDone ? 0 : 0;

  const reset = useCallback(() => {
    setElapsed(0);
    setAfterDone(false);
    setBeforeDone(false);
    setBeforeDots([]);
    setAfterDots([]);
    dotIdRef.current = 0;
  }, []);

  // Start/restart loop
  const startSimulation = useCallback(() => {
    reset();
    setRunning(true);
  }, [reset]);

  // IntersectionObserver to auto-start on scroll
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !hasStartedRef.current) {
          hasStartedRef.current = true;
          startSimulation();
        }
      },
      { threshold: 0.3 }
    );

    if (wrapperRef.current) {
      observerRef.current.observe(wrapperRef.current);
    }

    return () => {
      observerRef.current?.disconnect();
    };
  }, [startSimulation]);

  // Main tick
  useEffect(() => {
    if (!running) return;

    timerRef.current = setInterval(() => {
      setElapsed((prev) => {
        const next = prev + TICK_MS;

        if (next >= DURATION_AFTER_MS) setAfterDone(true);
        if (next >= DURATION_BEFORE_MS) setBeforeDone(true);

        // Both done → schedule reset
        if (next >= DURATION_BEFORE_MS) {
          setTimeout(() => {
            startSimulation();
          }, RESET_PAUSE_MS);
          setRunning(false);
          return next;
        }
        return next;
      });
    }, TICK_MS);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [running, startSimulation]);

  // Dot animation
  useEffect(() => {
    if (!running) return;

    const dotInterval = setInterval(() => {
      const spawnBefore = !beforeDone && Math.random() < 0.4;
      const spawnAfter = !afterDone && Math.random() < 0.65;

      if (spawnBefore) {
        setBeforeDots((prev) => {
          const dots = [...prev, { id: dotIdRef.current++, progress: 0 }];
          return dots.slice(-MAX_DOTS);
        });
      }
      if (spawnAfter) {
        setAfterDots((prev) => {
          const dots = [...prev, { id: dotIdRef.current++, progress: 0 }];
          return dots.slice(-MAX_DOTS);
        });
      }

      // Advance existing dots
      setBeforeDots((prev) =>
        prev
          .map((d) => ({ ...d, progress: d.progress + 0.08 }))
          .filter((d) => d.progress <= 1)
      );
      setAfterDots((prev) =>
        prev
          .map((d) => ({ ...d, progress: d.progress + 0.14 }))
          .filter((d) => d.progress <= 1)
      );
    }, 100);

    return () => clearInterval(dotInterval);
  }, [running, beforeDone, afterDone]);

  const renderFlowStack = (
    layers: { label: string; warning: boolean }[],
    dots: Dot[],
    isBefore: boolean
  ) => (
    <div className={styles.flowStack}>
      <div className={styles.dotsCanvas}>
        {dots.map((dot) => (
          <div
            key={dot.id}
            className={`${styles.dot} ${isBefore ? styles.dotSlow : styles.dotFast}`}
            style={{
              top: `${dot.progress * 92}%`,
              left: `${48 + Math.sin(dot.id * 1.7) * 12}%`,
              opacity: dot.progress > 0.85 ? (1 - dot.progress) * 6 : 0.8,
            }}
          />
        ))}
      </div>
      {layers.map((layer, i) => (
        <React.Fragment key={i}>
          {i > 0 && <div className={styles.flowArrow}>&darr;</div>}
          <div
            className={`${styles.flowLayer} ${layer.warning ? styles.flowLayerWarning : ''}`}
          >
            {layer.label}
            {layer.warning && <span className={styles.overheadTag}>&larr; overhead</span>}
          </div>
        </React.Fragment>
      ))}
    </div>
  );

  const formatNum = (n: number) => n.toLocaleString();

  return (
    <div className={styles.benchmarkWrapper} ref={wrapperRef}>
      <div className={styles.benchmarkConfig}>
        50,000 requests &middot; 1,000 concurrent &middot; 1 worker
      </div>

      <div className={styles.benchmarkColumns}>
        {/* Before column */}
        <div className={styles.benchmarkColumn}>
          <div className={`${styles.columnTitle} ${styles.columnTitleBefore}`}>
            Before (1 ASGI + 1 BaseHTTP)
            {beforeDone && (
              <span className={`${styles.doneBadge} ${styles.doneBadgeBefore}`}>done</span>
            )}
          </div>
          {renderFlowStack(BEFORE_LAYERS, beforeDots, true)}
          <div className={styles.statsRow}>
            <div className={styles.stat}>
              <div className={styles.statValue}>{formatNum(beforeCurrentRPS)}</div>
              <div className={styles.statLabel}>RPS</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statValue}>{formatNum(beforeCompleted)}</div>
              <div className={styles.statLabel}>Completed</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statValue}>{BEFORE_P50}ms</div>
              <div className={styles.statLabel}>P50</div>
            </div>
          </div>
          <div className={styles.progressBar}>
            <div
              className={`${styles.progressFill} ${styles.progressFillBefore}`}
              style={{ width: `${beforeProgress * 100}%` }}
            />
          </div>
        </div>

        {/* After column */}
        <div className={styles.benchmarkColumn}>
          <div className={`${styles.columnTitle} ${styles.columnTitleAfter}`}>
            After (2x Pure ASGI)
            {afterDone && (
              <span className={`${styles.doneBadge} ${styles.doneBadgeAfter}`}>done</span>
            )}
          </div>
          {renderFlowStack(AFTER_LAYERS, afterDots, false)}
          <div className={styles.statsRow}>
            <div className={styles.stat}>
              <div className={styles.statValue}>{formatNum(afterCurrentRPS)}</div>
              <div className={styles.statLabel}>RPS</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statValue}>{formatNum(afterCompleted)}</div>
              <div className={styles.statLabel}>Completed</div>
            </div>
            <div className={styles.stat}>
              <div className={styles.statValue}>{AFTER_P50}ms</div>
              <div className={styles.statLabel}>P50</div>
            </div>
          </div>
          <div className={styles.progressBar}>
            <div
              className={`${styles.progressFill} ${styles.progressFillAfter}`}
              style={{ width: `${afterProgress * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Summary stats */}
      <div className={styles.summaryStats}>
        <div className={styles.summaryItem}>
          <div className={styles.summaryValue}>+74%</div>
          <div className={styles.summaryLabel}>Throughput (RPS)</div>
        </div>
        <div className={styles.summaryItem}>
          <div className={styles.summaryValue}>-38%</div>
          <div className={styles.summaryLabel}>Median Latency (P50)</div>
        </div>
      </div>

      {/* Collapsible per-run data */}
      <div className={styles.collapsible}>
        <button
          className={styles.collapsibleToggle}
          onClick={() => setTableOpen(!tableOpen)}
        >
          <span
            className={`${styles.collapsibleChevron} ${
              tableOpen ? styles.collapsibleChevronOpen : ''
            }`}
          >
            &#9654;
          </span>
          Per-run data (3 runs each)
        </button>
        <div
          className={`${styles.collapsibleContent} ${
            tableOpen ? styles.collapsibleContentOpen : ''
          }`}
        >
          <table className={styles.dataTable}>
            <thead>
              <tr>
                <th>Config</th>
                <th>Run</th>
                <th>RPS</th>
                <th>P50 (ms)</th>
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_RUNS.map((row, i) => (
                <tr key={i}>
                  <td>{row.config}</td>
                  <td>{row.run}</td>
                  <td>{formatNum(row.rps)}</td>
                  <td>{row.p50}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
