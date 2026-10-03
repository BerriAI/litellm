"use client";

import { useEffect, useRef } from "react";

export const dotColors = ["#8b5cf6", "#22b3e8", "#e3a32b", "#eb6b93"] as const;
const lensBlue = { light: "#0011b3", dark: "#8b9bff" };
const columns = 120;
const rows = 7;
const cell = 6;

function dotColor(lit: boolean, pastLens: boolean, incoming: string, blue: string): string {
  if (!lit) return "#94a3b8";
  return pastLens ? blue : incoming;
}

export function DotFlow({ active }: { active: readonly string[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const node = canvas.current;
    const context = node?.getContext("2d");
    if (!node || !context) return;
    const colors = active.length ? active : ["#94a3b8"];
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const blue = document.documentElement.classList.contains("dark") ? lensBlue.dark : lensBlue.light;
    const lensColumn = Math.floor(columns * 0.62);
    const draw = (time: number) => {
      context.clearRect(0, 0, node.width, node.height);
      for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
          const x = column * cell + cell / 2;
          const y = row * cell + cell / 2;
          const center = (rows - 1) / 2;
          const funnel =
            column < lensColumn ? Math.abs(row - center) <= center * (1 - column / lensColumn) + 0.6 : row === center;
          const wave = Math.sin(column * 0.55 - time / 260 + row * 1.7);
          const lit = funnel && wave > 0.35;
          context.globalAlpha = lit ? 0.9 : 0.12;
          context.fillStyle = dotColor(lit, column >= lensColumn, colors[(row + column) % colors.length], blue);
          context.beginPath();
          context.arc(x, y, lit ? 1.6 : 1, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.globalAlpha = 1;
      context.strokeStyle = blue;
      context.lineWidth = 1.5;
      const lx = lensColumn * cell - 1;
      context.beginPath();
      context.moveTo(lx + 3, 1);
      context.lineTo(lx, 1);
      context.lineTo(lx, rows * cell - 1);
      context.lineTo(lx + 3, rows * cell - 1);
      context.stroke();
    };
    if (still) {
      draw(0);
      return;
    }
    let frame = requestAnimationFrame(function loop(time) {
      draw(time);
      frame = requestAnimationFrame(loop);
    });
    return () => cancelAnimationFrame(frame);
  }, [active]);
  return (
    <canvas
      ref={canvas}
      aria-hidden="true"
      width={columns * cell}
      height={rows * cell}
      className="h-10 w-full opacity-80"
    />
  );
}
