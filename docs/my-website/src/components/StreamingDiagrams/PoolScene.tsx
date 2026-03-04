import React, { useState, useEffect, useRef, useCallback } from 'react';
import styles from './styles.module.css';

/* ─── Layout ─── */
const W = 500;
const H = 300;
const CX = W / 2;

const SLOT_W = 60;
const SLOT_H = 44;
const SLOT_GAP = 28;
const NUM_SLOTS = 3;
const SLOTS_TOTAL = NUM_SLOTS * SLOT_W + (NUM_SLOTS - 1) * SLOT_GAP;
const SLOTS_X0 = (W - SLOTS_TOTAL) / 2;
const SLOTS_Y = 30;
const SLOT_BOTTOM = SLOTS_Y + SLOT_H;

const PROXY_W = 100;
const PROXY_H = 26;
const PROXY_Y = 145;
const PROXY_BOTTOM = PROXY_Y + PROXY_H;

const CLIENT_W = 50;
const CLIENT_H = 24;
const CLIENT_Y = 240;

const CONN_B_LABEL_X = W - 20;

const slotX = (i: number) => SLOTS_X0 + i * (SLOT_W + SLOT_GAP);
const slotCenterX = (i: number) => slotX(i) + SLOT_W / 2;

const CONN_B_MID_Y = (SLOT_BOTTOM + PROXY_Y) / 2;
const CONN_A_MID_Y = (PROXY_BOTTOM + CLIENT_Y) / 2;

/* ─── Types ─── */
type SlotColor = 'empty' | 'blue' | 'red' | 'green';

interface Step {
  duration: number;
  slots: SlotColor[];
  leaked: number[];
  fixed: number[];
  streaming: boolean;
  streamSlot: number;
  clientVisible: boolean;
  disconnected: boolean;
  shaking: boolean;
  showTimeout: boolean;
  dotsToProxy: boolean;
  hideLines: boolean;
  caption: string;
}

/* ─── Helpers ─── */
const defaults: Omit<Step, 'duration' | 'slots' | 'caption'> = {
  leaked: [], fixed: [], streaming: false, streamSlot: 0,
  clientVisible: false, disconnected: false, shaking: false, showTimeout: false,
  dotsToProxy: false, hideLines: false,
};

const step = (
  duration: number, slots: SlotColor[], caption: string,
  o: Partial<Step> = {},
): Step => ({ ...defaults, duration, slots, caption, ...o });

const EMPTY: SlotColor[] = ['empty', 'empty', 'empty'];
const ALL_RED: SlotColor[] = ['red', 'red', 'red'];

/* ═══ Timelines ═══ */

const HAPPY_STEPS: Step[] = [
  step(2000, ['blue', 'empty', 'empty'], 'Request arrives — connection acquired from pool', { clientVisible: true, streamSlot: 0 }),
  step(3000, ['blue', 'empty', 'empty'], 'Streaming response chunks…', { streaming: true, streamSlot: 0, clientVisible: true }),
  step(1500, ['green', 'empty', 'empty'], 'Stream complete — connection released', { clientVisible: true, streamSlot: 0 }),
  step(2000, EMPTY, 'Connection returned to pool ✓'),
  step(2000, EMPTY, '\u00A0'),
];

const LEAK_STEPS: Step[] = [
  // Client 1 — streams, disconnects, leaks
  step(1800, ['blue', 'empty', 'empty'], 'Request 1 — connection acquired', { clientVisible: true, streamSlot: 0 }),
  step(2000, ['blue', 'empty', 'empty'], 'Streaming…', { streaming: true, streamSlot: 0, clientVisible: true }),
  step(1800, ['red', 'empty', 'empty'], 'Client disconnects — slot 1 leaked', { clientVisible: true, disconnected: true, streamSlot: 0, leaked: [0] }),
  // Client 2 — arrive immediately
  step(1800, ['red', 'blue', 'empty'], 'Request 2 — connection acquired', { leaked: [0], clientVisible: true, streamSlot: 1 }),
  step(2000, ['red', 'blue', 'empty'], 'Streaming…', { leaked: [0], streaming: true, streamSlot: 1, clientVisible: true }),
  step(1800, ['red', 'red', 'empty'], 'Client disconnects — slot 2 leaked', { leaked: [0, 1], clientVisible: true, disconnected: true, streamSlot: 1 }),
  // Client 3
  step(1800, ['red', 'red', 'blue'], 'Request 3 — connection acquired', { leaked: [0, 1], clientVisible: true, streamSlot: 2 }),
  step(2000, ['red', 'red', 'blue'], 'Streaming…', { leaked: [0, 1], streaming: true, streamSlot: 2, clientVisible: true }),
  step(1800, ALL_RED, 'Client disconnects — all slots leaked', { leaked: [0, 1, 2], clientVisible: true, disconnected: true, streamSlot: 2 }),
  // Pool exhausted — dots hit proxy but nothing connects beyond
  step(3000, ALL_RED, 'New request — pool exhausted, nowhere to go', { leaked: [0, 1, 2], dotsToProxy: true }),
  step(2000, EMPTY, '\u00A0'),
];

const FIX_STEPS: Step[] = [
  step(1800, ['empty', 'blue', 'empty'], 'Streaming request', { clientVisible: true, streamSlot: 1 }),
  step(2500, ['empty', 'blue', 'empty'], 'Streaming…', { streaming: true, streamSlot: 1, clientVisible: true }),
  step(2000, ['empty', 'blue', 'empty'], 'Client disconnects (Connection A)', { clientVisible: true, disconnected: true, streamSlot: 1 }),
  step(2000, ['empty', 'green', 'empty'], 'finally block runs — Connection B released', { fixed: [1], clientVisible: true, streamSlot: 1, hideLines: true }),
  step(2000, EMPTY, 'Slot freed. Pool stays healthy. ✓', { clientVisible: true, streamSlot: 1, hideLines: true }),
];

/* ═══ Renderer ═══ */

function PoolScene({ steps }: { steps: Step[] }) {
  const [stepIdx, setStepIdx] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) { clearTimeout(timerRef.current); timerRef.current = null; }
  }, []);

  useEffect(() => {
    const s = steps[stepIdx];
    const delay = stepIdx === steps.length - 1 ? s.duration + 1000 : s.duration;
    timerRef.current = setTimeout(() => setStepIdx((p) => (p + 1) % steps.length), delay);
    return clearTimer;
  }, [stepIdx, steps, clearTimer]);

  const s = steps[stepIdx];
  const slotAnchorX = slotCenterX(s.streamSlot);
  const showChain = (s.clientVisible || s.streaming) && !s.hideLines;

  return (
    <div className={styles.sceneWrapper}>
      <svg className={styles.sceneSvg} viewBox={`0 0 ${W} ${H}`}>

        {/* ── Header ── */}
        <text x={CX} y={14} textAnchor="middle" fill="var(--pg-text-secondary)" fontSize={11} fontWeight={700}>
          Connection Pool
        </text>

        {/* ── Slots ── */}
        {s.slots.map((color, i) => {
          const sx = slotX(i);
          const cx = slotCenterX(i);
          const empty = color === 'empty';
          const fill = empty ? 'transparent'
            : color === 'blue' ? '#3b82f6'
            : color === 'red' ? '#ef4444'
            : '#22c55e';
          const stroke = empty ? 'var(--pg-spot-border)' : fill;
          return (
            <g key={i}>
              <rect
                x={sx} y={SLOTS_Y} width={SLOT_W} height={SLOT_H} rx={8}
                fill={fill} stroke={stroke} strokeWidth={1.5}
                style={{ transition: 'fill 0.4s, stroke 0.4s' }}
              />
              <text x={cx} y={SLOTS_Y + SLOT_H + 14} textAnchor="middle"
                fill="var(--pg-text-dim)" fontSize={10} fontFamily="var(--pg-mono)">
                {i + 1}
              </text>

              {s.leaked.includes(i) && (
                <g className={styles.pulse}>
                  <rect x={cx - 24} y={SLOTS_Y + SLOT_H + 20} width={48} height={14} rx={3} fill="#ef4444" />
                  <text x={cx} y={SLOTS_Y + SLOT_H + 30.5} textAnchor="middle"
                    fill="#fff" fontSize={8} fontWeight={800} letterSpacing="0.5px">
                    LEAKED
                  </text>
                </g>
              )}

              {s.fixed.includes(i) && (
                <g>
                  <rect x={cx - 34} y={SLOTS_Y + SLOT_H + 20} width={68} height={14} rx={3} fill="#22c55e" />
                  <text x={cx} y={SLOTS_Y + SLOT_H + 30.5} textAnchor="middle"
                    fill="#fff" fontSize={7.5} fontWeight={800} letterSpacing="0.5px">
                    RELEASED ✓
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* ── Connection B: slot → proxy ── */}
        {showChain && (
          <g>
            <line
              x1={slotAnchorX} y1={SLOT_BOTTOM}
              x2={CX} y2={PROXY_Y}
              stroke="var(--pg-spot-border)" strokeWidth={1} strokeDasharray="4 3" opacity={0.4}
            />
            <text x={CONN_B_LABEL_X} y={CONN_B_MID_Y + 4} textAnchor="end"
              fill="var(--pg-text-dim)" fontSize={8} fontFamily="var(--pg-mono)">
              Connection B
            </text>
          </g>
        )}

        {/* ── Connection A: proxy → client ── */}
        {showChain && (
          <g>
            <line
              x1={CX} y1={PROXY_BOTTOM}
              x2={CX} y2={CLIENT_Y}
              stroke="var(--pg-spot-border)" strokeWidth={1} strokeDasharray="4 3" opacity={0.4}
            />
            <text x={CONN_B_LABEL_X} y={CONN_A_MID_Y + 4} textAnchor="end"
              fill="var(--pg-text-dim)" fontSize={8} fontFamily="var(--pg-mono)">
              Connection A
            </text>
          </g>
        )}

        {/* ── Streaming dots on Connection B (slot → proxy) ── */}
        {s.streaming && (
          <g>
            {[0, 1, 2].map((i) => (
              <circle key={`b${i}`} r={3} fill="var(--pg-dot-color)"
                className={styles.streamDot}
                style={{
                  '--dot-from-x': `${slotAnchorX}px`,
                  '--dot-from-y': `${SLOT_BOTTOM + 4}px`,
                  '--dot-to-x': `${CX}px`,
                  '--dot-to-y': `${PROXY_Y - 4}px`,
                  animationDelay: `${i * 0.4}s`,
                } as React.CSSProperties}
              />
            ))}
          </g>
        )}

        {/* ── Streaming dots on Connection A (proxy → client, delayed) ── */}
        {s.streaming && (
          <g>
            {[0, 1, 2].map((i) => (
              <circle key={`a${i}`} r={3} fill="var(--pg-dot-color)"
                className={styles.streamDot}
                style={{
                  '--dot-from-x': `${CX}px`,
                  '--dot-from-y': `${PROXY_BOTTOM + 4}px`,
                  '--dot-to-x': `${CX}px`,
                  '--dot-to-y': `${CLIENT_Y - 4}px`,
                  animationDelay: `${1.5 + i * 0.4}s`,
                } as React.CSSProperties}
              />
            ))}
          </g>
        )}

        {/* ── Dots hitting proxy with nowhere to go (pool exhausted) ── */}
        {s.dotsToProxy && (
          <g>
            {/* Proxy box */}
            <rect
              x={CX - PROXY_W / 2} y={PROXY_Y}
              width={PROXY_W} height={PROXY_H} rx={6}
              fill="var(--pg-caption-bg)" stroke="var(--pg-spot-border)" strokeWidth={1}
            />
            <text x={CX} y={PROXY_Y + PROXY_H / 2 + 4} textAnchor="middle"
              fill="var(--pg-text-primary)" fontSize={10} fontWeight={600} fontFamily="var(--pg-mono)">
              LiteLLM Proxy
            </text>
            {/* Client below */}
            <rect
              x={CX - CLIENT_W / 2} y={CLIENT_Y}
              width={CLIENT_W} height={CLIENT_H} rx={6}
              fill="none" stroke="var(--pg-spot-border)" strokeWidth={1}
            />
            <text x={CX} y={CLIENT_Y + CLIENT_H / 2 + 4} textAnchor="middle"
              fill="var(--pg-text-secondary)" fontSize={10} fontFamily="var(--pg-mono)">
              Client
            </text>
            {/* Connection A line */}
            <line
              x1={CX} y1={PROXY_BOTTOM}
              x2={CX} y2={CLIENT_Y}
              stroke="var(--pg-spot-border)" strokeWidth={1} strokeDasharray="4 3" opacity={0.4}
            />
            {/* Dots coming from client up to proxy — but no connection B above */}
            {[0, 1, 2].map((i) => (
              <circle key={`e${i}`} r={3} fill="var(--pg-dot-color)"
                className={styles.streamDot}
                style={{
                  '--dot-from-x': `${CX}px`,
                  '--dot-from-y': `${CLIENT_Y - 4}px`,
                  '--dot-to-x': `${CX}px`,
                  '--dot-to-y': `${PROXY_Y + PROXY_H + 4}px`,
                  animationDelay: `${i * 0.4}s`,
                } as React.CSSProperties}
              />
            ))}
            {/* Question mark above proxy — no connection to pool */}
            <text x={CX} y={PROXY_Y - 12} textAnchor="middle"
              fill="var(--pg-text-dim)" fontSize={16} fontWeight={700}
              className={styles.pulse}>
              ?
            </text>
          </g>
        )}

        {/* ── LiteLLM Proxy box (centered) ── */}
        {(showChain || s.clientVisible) && !s.dotsToProxy && (
          <g>
            <rect
              x={CX - PROXY_W / 2} y={PROXY_Y}
              width={PROXY_W} height={PROXY_H} rx={6}
              fill="var(--pg-caption-bg)" stroke="var(--pg-spot-border)" strokeWidth={1}
            />
            <text x={CX} y={PROXY_Y + PROXY_H / 2 + 4} textAnchor="middle"
              fill="var(--pg-text-primary)" fontSize={10} fontWeight={600} fontFamily="var(--pg-mono)">
              LiteLLM Proxy
            </text>
          </g>
        )}

        {/* ── Client box (centered) ── */}
        {s.clientVisible && !s.dotsToProxy && (
          <g>
            <rect
              x={CX - CLIENT_W / 2} y={CLIENT_Y}
              width={CLIENT_W} height={CLIENT_H} rx={6}
              fill="none" stroke="var(--pg-spot-border)" strokeWidth={1}
            />
            <text x={CX} y={CLIENT_Y + CLIENT_H / 2 + 4} textAnchor="middle"
              fill="var(--pg-text-secondary)" fontSize={10} fontFamily="var(--pg-mono)">
              Client
            </text>
          </g>
        )}

        {/* ── Disconnect ✕ on Connection A ── */}
        {s.disconnected && (
          <g>
            <circle cx={CX} cy={CONN_A_MID_Y} r={16} fill="#ef4444" opacity={0.15} />
            <text x={CX} y={CONN_A_MID_Y + 1} textAnchor="middle" dominantBaseline="central"
              fill="#ef4444" fontSize={20} fontWeight={900}>
              ✕
            </text>
          </g>
        )}

        {/* ── Shaking blocked request ── */}
        {s.shaking && (
          <g>
            <rect width={16} height={16} rx={3} fill="#3b82f6"
              x={SLOTS_X0 - 30} y={SLOTS_Y + SLOT_H / 2 - 8}
              className={styles.shake}
            />
            <text x={SLOTS_X0 - 22} y={SLOTS_Y + SLOT_H / 2 - 14} textAnchor="middle"
              fill="var(--pg-text-secondary)" fontSize={16} fontWeight={700}
              className={styles.pulse}>
              ?
            </text>
          </g>
        )}

      </svg>

      {/* ── Timeout overlay ── */}
      {s.showTimeout && (
        <div className={styles.timeoutOverlay}>
          <span className={`${styles.timeoutText} ${styles.pulse}`}>TIMEOUT</span>
          <span className={styles.timeoutSubtext}>pool exhausted</span>
        </div>
      )}

      <div className={styles.sceneCaption}>{s.caption}</div>
    </div>
  );
}

/* ═══ Exports ═══ */
export function HappyPathScene() { return <PoolScene steps={HAPPY_STEPS} />; }
export function LeakScene() { return <PoolScene steps={LEAK_STEPS} />; }
export function FixScene() { return <PoolScene steps={FIX_STEPS} />; }
