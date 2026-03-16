import React from 'react';

const W = 720;
const HALF_W = W / 2 - 12;
const BOX_H = 34;
const BOX_RX = 6;
const GAP_Y = 12;
const CODE_LINE_H = 16;
const CODE_FONT = 10;
const LABEL_FONT = 10.5;
const SUB_FONT = 9;
const MONO = "var(--pg-mono)";

/* ─── Code block ─── */
function CodeBlock({ x, y, width, lines }: {
  x: number; y: number; width: number;
  lines: { text: string; color?: string; highlight?: string }[];
}) {
  const blockH = lines.length * CODE_LINE_H + 8;
  return (
    <g>
      <rect x={x} y={y} width={width} height={blockH} rx={6}
        fill="var(--pg-spot-bg)" stroke="var(--pg-spot-border)" strokeWidth={1} />
      {lines.map((line, i) => (
        <g key={i}>
          {line.highlight && (
            <rect x={x + 1} y={y + 4 + i * CODE_LINE_H - 1}
              width={width - 2} height={CODE_LINE_H}
              fill={line.highlight} opacity={0.12} rx={2} />
          )}
          <text x={x + 8} y={y + 4 + i * CODE_LINE_H + CODE_LINE_H * 0.72}
            fill={line.color || 'var(--pg-text-secondary)'}
            fontSize={CODE_FONT} fontFamily={MONO} fontWeight={line.color ? 600 : 400}>
            {line.text}
          </text>
        </g>
      ))}
    </g>
  );
}

/* ─── Step box with label + subtitle ─── */
function StepBox({ cx, y, width, label, subtitle, status }: {
  cx: number; y: number; width: number;
  label: string; subtitle?: string;
  status: 'normal' | 'error' | 'success';
}) {
  const boxX = cx - width / 2;
  const h = subtitle ? BOX_H + 10 : BOX_H;

  const fill = status === 'error' ? 'rgba(239, 68, 68, 0.08)'
    : status === 'success' ? 'rgba(34, 197, 94, 0.08)'
    : 'var(--pg-spot-bg)';
  const stroke = status === 'error' ? 'rgba(239, 68, 68, 0.4)'
    : status === 'success' ? 'rgba(34, 197, 94, 0.4)'
    : 'var(--pg-spot-border)';
  const textColor = status === 'error' ? '#ef4444'
    : status === 'success' ? '#22c55e'
    : 'var(--pg-text-primary)';

  return (
    <g>
      <rect x={boxX} y={y} width={width} height={h} rx={BOX_RX}
        fill={fill} stroke={stroke} strokeWidth={1.5} />
      <text x={cx} y={subtitle ? y + h / 2 - 5 : y + h / 2 + 1}
        textAnchor="middle" dominantBaseline="central"
        fill={textColor} fontSize={LABEL_FONT} fontFamily={MONO} fontWeight={600}>
        {label}
      </text>
      {subtitle && (
        <text x={cx} y={y + h / 2 + 9}
          textAnchor="middle" dominantBaseline="central"
          fill={status === 'error' ? 'rgba(239, 68, 68, 0.7)' : status === 'success' ? 'rgba(34, 197, 94, 0.7)' : 'var(--pg-text-dim)'}
          fontSize={SUB_FONT} fontFamily={MONO}>
          {subtitle}
        </text>
      )}
    </g>
  );
}

/* ─── Arrow connector ─── */
function DownArrow({ cx, y1, y2 }: { cx: number; y1: number; y2: number }) {
  return (
    <g>
      <line x1={cx} y1={y1} x2={cx} y2={y2 - 5}
        stroke="var(--pg-spot-border)" strokeWidth={1} strokeDasharray="3 3" />
      <polygon
        points={`${cx},${y2} ${cx - 3.5},${y2 - 5} ${cx + 3.5},${y2 - 5}`}
        fill="var(--pg-spot-border)" />
    </g>
  );
}

/* ─── Column ─── */
function Column({ offsetX, title, titleColor, code, steps, result, resultColor }: {
  offsetX: number;
  title: string;
  titleColor: string;
  code: { text: string; color?: string; highlight?: string }[];
  steps: { label: string; subtitle?: string; status: 'normal' | 'error' | 'success' }[];
  result: string;
  resultColor: string;
}) {
  const colW = HALF_W;
  const cx = offsetX + colW / 2;
  const boxW = colW - 40;

  let y = 16;
  const titleY = y; y += 26;
  const codeY = y;
  const codeH = code.length * CODE_LINE_H + 8;
  y += codeH + GAP_Y + 4;

  const stepPositions: number[] = [];
  for (const step of steps) {
    const h = step.subtitle ? BOX_H + 10 : BOX_H;
    if (stepPositions.length > 0) y += GAP_Y;
    stepPositions.push(y);
    y += h;
  }

  const resultY = y + 18;

  return (
    <g>
      <text x={cx} y={titleY} textAnchor="middle" fill={titleColor}
        fontSize={12} fontWeight={700} fontFamily={MONO}>
        {title}
      </text>

      <CodeBlock x={offsetX + 12} y={codeY} width={colW - 24} lines={code} />

      {steps.map((step, i) => {
        const sy = stepPositions[i];
        const prevH = i > 0 ? (steps[i - 1].subtitle ? BOX_H + 10 : BOX_H) : 0;
        return (
          <g key={i}>
            {i === 0 && <DownArrow cx={cx} y1={codeY + codeH} y2={sy} />}
            {i > 0 && <DownArrow cx={cx} y1={stepPositions[i - 1] + prevH} y2={sy} />}
            <StepBox cx={cx} y={sy} width={boxW}
              label={step.label} subtitle={step.subtitle} status={step.status} />
          </g>
        );
      })}

      <text x={cx} y={resultY} textAnchor="middle"
        fill={resultColor} fontSize={11} fontWeight={700} fontFamily={MONO}>
        {result}
      </text>
    </g>
  );
}

/* ─── Main ─── */
export function ContentVsStreamDiagram() {
  const H = 310;
  const dividerX = W / 2;

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <Column
          offsetX={0}
          title="content= (before)"
          titleColor="#ef4444"
          code={[
            { text: 'httpx.Response(' },
            { text: '    content=stream,', color: '#ef4444', highlight: '#ef4444' },
            { text: ')' },
          ]}
          steps={[
            { label: 'Data consumed into memory', subtitle: 'Original stream discarded', status: 'normal' },
            { label: 'aclose() has nothing to close', subtitle: 'Points at an empty wrapper', status: 'error' },
            { label: 'Connection stuck in pool', status: 'error' },
          ]}
          result="Leaked on every request"
          resultColor="#ef4444"
        />

        <line x1={dividerX} y1={8} x2={dividerX} y2={H - 8}
          stroke="var(--pg-spot-border)" strokeWidth={1} opacity={0.5} />

        <Column
          offsetX={dividerX + 12}
          title="stream= (after)"
          titleColor="#22c55e"
          code={[
            { text: 'httpx.Response(' },
            { text: '    stream=stream,', color: '#22c55e', highlight: '#22c55e' },
            { text: ')' },
          ]}
          steps={[
            { label: 'Original stream preserved', subtitle: 'Data flows through on demand', status: 'normal' },
            { label: 'aclose() reaches real connection', subtitle: 'Propagates through to aiohttp', status: 'success' },
            { label: 'Connection returned to pool', status: 'success' },
          ]}
          result="Released ✓"
          resultColor="#22c55e"
        />
      </svg>
    </div>
  );
}
