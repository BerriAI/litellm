export const hashN = (n: number, k: number): number => {
  const v = Math.sin(n * 127.1 + k * 311.7) * 43758.5453;
  return v - Math.floor(v);
};

export const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

type Draw = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void;

function canvasLoop(canvas: HTMLCanvasElement, draw: Draw): () => void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return () => {};
  const reduce = prefersReducedMotion();
  let w = 0;
  let h = 0;
  let rafId = 0;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    w = rect.width;
    h = rect.height;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const frame = (ms: number) => {
    ctx.clearRect(0, 0, w, h);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    draw(ctx, w, h, ms / 1000);
    if (!reduce) rafId = requestAnimationFrame(frame);
  };

  resize();
  const observer = new ResizeObserver(() => {
    resize();
    if (reduce) frame(0);
  });
  observer.observe(canvas);
  if (reduce) frame(0);
  else rafId = requestAnimationFrame(frame);
  let visibilityObserver: IntersectionObserver | null = null;
  if (typeof IntersectionObserver !== "undefined") {
    visibilityObserver = new IntersectionObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      if (entry.isIntersecting) {
        if (!reduce) {
          cancelAnimationFrame(rafId);
          rafId = requestAnimationFrame(frame);
        }
      } else {
        cancelAnimationFrame(rafId);
      }
    });
    visibilityObserver.observe(canvas);
  }
  return () => {
    cancelAnimationFrame(rafId);
    observer.disconnect();
    visibilityObserver?.disconnect();
  };
}

const STAR_COLORS = ["#cfe3ff", "#9cc3ff", "#ffffff", "#ffb98a", "#8b9bff", "#e9d8ff"];

export function startStarfield(canvas: HTMLCanvasElement): () => void {
  let stars: { x: number; y: number; r: number; color: string; phase: number; depth: number }[] = [];
  let seededFor = "";
  return canvasLoop(canvas, (ctx, w, h, t) => {
    const key = `${Math.round(w)}x${Math.round(h)}`;
    if (key !== seededFor) {
      seededFor = key;
      stars = Array.from({ length: Math.round((w * h) / 5200) }, (_, n) => ({
        x: hashN(n, 1) * w,
        y: hashN(n, 2) * h,
        r: hashN(n, 3) < 0.06 ? 1.5 : 0.4 + hashN(n, 4) * 0.8,
        color: STAR_COLORS[Math.floor(hashN(n, 5) * STAR_COLORS.length)],
        phase: hashN(n, 6) * Math.PI * 2,
        depth: 0.2 + hashN(n, 7) * 0.8,
      }));
    }
    for (const s of stars) {
      const y = (((s.y - t * 3 * s.depth) % h) + h) % h;
      ctx.globalAlpha = (0.25 + 0.5 * (0.5 + 0.5 * Math.sin(t * 0.8 + s.phase))) * s.depth;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(s.x, y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

const CITY = Array.from({ length: 700 }, (_, n) => ({
  a: (hashN(n, 1) - 0.5) * 1.3,
  depth: Math.pow(hashN(n, 2), 2) * 0.08,
  r: 0.4 + hashN(n, 3) * 0.9,
  twinkle: hashN(n, 4) * Math.PI * 2,
}));

export const PLANET_HORIZON = 0.8;

export function startPlanetrise(canvas: HTMLCanvasElement): () => void {
  return canvasLoop(canvas, (ctx, w, h, t) => {
    const r = Math.max(w * 1.25, h * 1.4);
    const cx = w / 2;
    const cy = h * PLANET_HORIZON + r;
    const sunY = cy - r - h * (0.02 + 0.012 * Math.sin(t * 0.15));

    const sky = ctx.createRadialGradient(cx, sunY, 0, cx, sunY, w * 0.7);
    sky.addColorStop(0, "rgba(120,170,255,0.32)");
    sky.addColorStop(0.35, "rgba(70,110,220,0.1)");
    sky.addColorStop(1, "rgba(40,60,140,0)");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, w, h);

    ctx.globalCompositeOperation = "lighter";
    const flare = ctx.createLinearGradient(0, sunY, w, sunY);
    flare.addColorStop(0, "rgba(120,170,255,0)");
    flare.addColorStop(0.5, "rgba(210,230,255,0.55)");
    flare.addColorStop(1, "rgba(120,170,255,0)");
    ctx.fillStyle = flare;
    ctx.fillRect(0, sunY - 1.2, w, 2.4);
    const sun = ctx.createRadialGradient(cx, sunY, 0, cx, sunY, h * 0.16);
    sun.addColorStop(0, "rgba(255,255,255,1)");
    sun.addColorStop(0.08, "rgba(220,235,255,0.9)");
    sun.addColorStop(0.35, "rgba(140,180,255,0.22)");
    sun.addColorStop(1, "rgba(100,140,255,0)");
    ctx.fillStyle = sun;
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = "source-over";

    ctx.fillStyle = "#04060c";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    const limb = ctx.createRadialGradient(cx, cy, r * 0.985, cx, cy, r * 1.012);
    limb.addColorStop(0, "rgba(60,110,255,0)");
    limb.addColorStop(0.55, "rgba(110,160,255,0.55)");
    limb.addColorStop(0.75, "rgba(200,225,255,0.9)");
    limb.addColorStop(1, "rgba(120,160,255,0)");
    ctx.globalCompositeOperation = "lighter";
    ctx.fillStyle = limb;
    ctx.beginPath();
    ctx.arc(cx, cy, r * 1.012, Math.PI * 1.08, Math.PI * 1.92);
    ctx.arc(cx, cy, r * 0.985, Math.PI * 1.92, Math.PI * 1.08, true);
    ctx.closePath();
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";

    for (const c of CITY) {
      const theta = -Math.PI / 2 + c.a * (w / r);
      const rr = r * (0.996 - c.depth);
      const x = cx + Math.cos(theta) * rr;
      const y = cy + Math.sin(theta) * rr;
      if (y > h) continue;
      ctx.globalAlpha = (0.25 + 0.5 * (0.5 + 0.5 * Math.sin(t * 1.3 + c.twinkle))) * (1 - c.depth * 8);
      ctx.fillStyle = "#ffd59a";
      ctx.beginPath();
      ctx.arc(x, y, c.r, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}
