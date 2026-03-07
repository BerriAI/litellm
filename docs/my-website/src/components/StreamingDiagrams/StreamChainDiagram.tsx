import React from 'react';

/* ─── Layout constants ─── */
const W = 720;
const H = 240;
const BOX_H = 36;
const BOX_RX = 8;
const ARROW_GAP = 32;
const CODE_OFFSET_Y = 20;
const CODE_LINE_H = 18;
const CODE_FONT = 11;
const LABEL_FONT = 11;

interface Box {
  label: string;
  width: number;
  codeLines?: string[];
}

const boxes: Box[] = [
  { label: 'Provider HTTP Stream', width: 160 },
  {
    label: 'CustomStreamWrapper',
    width: 170,
    codeLines: [
      '+ async def aclose(self):',
      '+     await self.response.aclose()',
    ],
  },
  {
    label: 'async_data_generator',
    width: 170,
    codeLines: [
      'try:',
      '    async for chunk in stream:',
      '        yield chunk',
      '+ finally:',
      '+     await stream.aclose()',
    ],
  },
  { label: 'Client', width: 70 },
];

/* ─── Compute positions ─── */
const totalW = boxes.reduce((s, b) => s + b.width, 0) + (boxes.length - 1) * ARROW_GAP;
const startX = (W - totalW) / 2;

interface BoxPos {
  x: number;
  cx: number;
  y: number;
  w: number;
}

const positions: BoxPos[] = [];
let cx = startX;
const BOX_Y = 24;
for (const b of boxes) {
  positions.push({ x: cx, cx: cx + b.width / 2, y: BOX_Y, w: b.width });
  cx += b.width + ARROW_GAP;
}

/* ─── Arrow between boxes (left to right data flow) ─── */
function Arrow({ fromIdx, toIdx }: { fromIdx: number; toIdx: number }) {
  const from = positions[fromIdx];
  const to = positions[toIdx];
  const x1 = from.x + from.w;
  const x2 = to.x;
  const y = BOX_Y + BOX_H / 2;
  return (
    <g>
      <line x1={x1} y1={y} x2={x2 - 8} y2={y}
        stroke="var(--pg-text-dim)" strokeWidth={1.5} strokeDasharray="4 3" />
      {/* arrowhead pointing right */}
      <polygon
        points={`${x2},${y} ${x2 - 8},${y - 4} ${x2 - 8},${y + 4}`}
        fill="var(--pg-text-dim)"
      />
    </g>
  );
}

/* ─── Code diff block ─── */
function CodeBlock({ cx, lines, topY }: { cx: number; lines: string[]; topY: number }) {
  const maxLen = Math.max(...lines.map(l => l.length));
  const charW = CODE_FONT * 0.62;
  const blockW = maxLen * charW + 16;
  const blockH = lines.length * CODE_LINE_H + 8;
  const blockX = cx - blockW / 2;

  return (
    <g>
      <rect
        x={blockX} y={topY}
        width={blockW} height={blockH}
        rx={6}
        fill="var(--pg-spot-bg)"
        stroke="var(--pg-spot-border)"
        strokeWidth={1}
      />
      {lines.map((line, i) => {
        const isAdded = line.startsWith('+');
        return (
          <g key={i}>
            {isAdded && (
              <rect
                x={blockX + 1} y={topY + 4 + i * CODE_LINE_H - 1}
                width={blockW - 2} height={CODE_LINE_H}
                fill="#22c55e" opacity={0.12} rx={2}
              />
            )}
            <text
              x={blockX + 8}
              y={topY + 4 + i * CODE_LINE_H + CODE_LINE_H * 0.72}
              fill={isAdded ? '#22c55e' : 'var(--pg-text-secondary)'}
              fontSize={CODE_FONT}
              fontFamily="var(--pg-mono)"
              fontWeight={isAdded ? 600 : 400}
            >
              {line}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/* ─── Main component ─── */
export function StreamChainDiagram() {
  const codeTopY = BOX_Y + BOX_H + CODE_OFFSET_Y;

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        {/* Boxes */}
        {boxes.map((b, i) => {
          const p = positions[i];
          const isEndpoint = i === 0 || i === boxes.length - 1;
          return (
            <g key={i}>
              <rect
                x={p.x} y={p.y}
                width={p.w} height={BOX_H}
                rx={BOX_RX}
                fill={isEndpoint ? 'transparent' : 'var(--pg-caption-bg)'}
                stroke="var(--pg-spot-border)"
                strokeWidth={1.5}
              />
              <text
                x={p.cx} y={p.y + BOX_H / 2 + 1}
                textAnchor="middle" dominantBaseline="central"
                fill="var(--pg-text-primary)"
                fontSize={LABEL_FONT}
                fontFamily="var(--pg-mono)"
                fontWeight={600}
              >
                {b.label}
              </text>
            </g>
          );
        })}

        {/* Arrows (data flows left to right: Provider → CSW → generator → Client) */}
        <Arrow fromIdx={0} toIdx={1} />
        <Arrow fromIdx={1} toIdx={2} />
        <Arrow fromIdx={2} toIdx={3} />

        {/* Code diffs below relevant boxes */}
        {boxes.map((b, i) => {
          if (!b.codeLines) return null;
          return (
            <CodeBlock
              key={i}
              cx={positions[i].cx}
              lines={b.codeLines}
              topY={codeTopY}
            />
          );
        })}

        {/* Connector lines from boxes to code blocks */}
        {boxes.map((b, i) => {
          if (!b.codeLines) return null;
          const p = positions[i];
          return (
            <line
              key={`conn-${i}`}
              x1={p.cx} y1={p.y + BOX_H}
              x2={p.cx} y2={codeTopY}
              stroke="var(--pg-spot-border)"
              strokeWidth={1}
              strokeDasharray="3 3"
              opacity={0.5}
            />
          );
        })}
      </svg>
    </div>
  );
}
