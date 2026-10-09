"use client";

import React, { useEffect, useRef, useState } from "react";
import { ArrowRight, ArrowUpRight, Github, Play, Plug, Star } from "lucide-react";
import moyaiHead from "../../../public/assets/moyai/moyai-head.svg";
import anthropicLogo from "../../../public/assets/moyai/logos/anthropic.svg";
import bedrockLogo from "../../../public/assets/moyai/logos/bedrock.svg";
import deepseekLogo from "../../../public/assets/moyai/logos/deepseek.svg";
import fireworksLogo from "../../../public/assets/moyai/logos/fireworks.svg";
import googleLogo from "../../../public/assets/moyai/logos/google.svg";
import hermesLogo from "../../../public/assets/moyai/logos/hermes.png";
import langchainLogo from "../../../public/assets/moyai/logos/langchain.svg";
import mistralLogo from "../../../public/assets/moyai/logos/mistral.svg";
import openaiLogo from "../../../public/assets/moyai/logos/openai.svg";
import opencodeLogo from "../../../public/assets/moyai/logos/opencode.svg";
import xaiLogo from "../../../public/assets/moyai/logos/xai.svg";
import { PLANET_HORIZON, prefersReducedMotion, startPlanetrise, startStarfield } from "./moyaiSky";
import { normalizeMoyaiUrl } from "./moyaiConnect";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import styles from "./MoyaiLanding.module.css";

export const MOYAI_GITHUB_URL = "https://github.com/BerriAI/moyai";
export const MOYAI_WALKTHROUGH_URL = `${MOYAI_GITHUB_URL}#see-it-in-action`;
export const MOYAI_LAUNCH_POST_URL = "https://docs.litellm.ai/blog/moyai-open-source";
const MOYAI_DEMO_GIF_URL = "https://github.com/user-attachments/assets/2de74e6a-c37c-48d6-8a99-de2166c88626";

interface LogoItem {
  name: string;
  logo: { src: string };
}

const HARNESSES: LogoItem[] = [
  { name: "Claude Code", logo: anthropicLogo },
  { name: "Codex", logo: openaiLogo },
  { name: "Hermes", logo: hermesLogo },
  { name: "OpenCode", logo: opencodeLogo },
  { name: "Deep Agents", logo: langchainLogo },
];

const PROVIDERS: LogoItem[] = [
  { name: "OpenAI", logo: openaiLogo },
  { name: "Anthropic", logo: anthropicLogo },
  { name: "Fireworks", logo: fireworksLogo },
  { name: "Google", logo: googleLogo },
  { name: "xAI", logo: xaiLogo },
  { name: "Mistral", logo: mistralLogo },
  { name: "DeepSeek", logo: deepseekLogo },
  { name: "Bedrock", logo: bedrockLogo },
];

const ORBIT_TILT = 0.2;
const INNER_RADIUS: [number, number] = [0.16, 240];
const OUTER_RADIUS: [number, number] = [0.28, 420];

function useCanvasScene(start: (canvas: HTMLCanvasElement) => () => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    return start(ref.current);
  }, [start]);
  return ref;
}

function Orbit({
  items,
  radius,
  speed,
  heroRef,
  anchorRef,
  delay,
  className = "",
}: {
  items: LogoItem[];
  radius: [number, number];
  speed: number;
  heroRef: React.RefObject<HTMLElement | null>;
  anchorRef: React.RefObject<HTMLElement | null>;
  delay?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const reduce = prefersReducedMotion();
    let rafId = 0;
    const frame = (ms: number) => {
      const hero = heroRef.current?.getBoundingClientRect();
      const anchor = anchorRef.current?.getBoundingClientRect();
      if (hero && anchor) {
        const t = ms / 1000;
        const top = anchor.bottom - hero.top + 24;
        const bottom = hero.height * PLANET_HORIZON - 24;
        const outer = Math.min(hero.width * OUTER_RADIUS[0], OUTER_RADIUS[1]);
        const scale = Math.min(1, Math.max(0, (bottom - top) / 2 - 18) / (outer * ORBIT_TILT));
        const rx = Math.min(hero.width * radius[0], radius[1]) * scale;
        const ry = rx * ORBIT_TILT;
        el.style.top = `${(top + bottom) / 2}px`;
        el.dataset.placed = "true";
        Array.from(el.children).forEach((child, i) => {
          const chip = child as HTMLElement;
          const theta = (i / items.length) * Math.PI * 2 + t * speed;
          const front = Math.sin(theta) > 0;
          chip.style.transform = `translate(${Math.cos(theta) * rx}px, ${Math.sin(theta) * ry}px) translate(-50%, -50%)`;
          chip.style.zIndex = front ? "2" : "1";
          chip.style.opacity = front ? "1" : "0.55";
        });
      }
      if (!reduce) rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);
    let resizeObserver: ResizeObserver | null = null;
    let onResize: (() => void) | null = null;
    if (reduce) {
      const reposition = () => frame(performance.now());
      if (typeof ResizeObserver !== "undefined" && heroRef.current) {
        resizeObserver = new ResizeObserver(reposition);
        resizeObserver.observe(heroRef.current);
      } else {
        onResize = reposition;
        window.addEventListener("resize", reposition);
      }
    }
    return () => {
      cancelAnimationFrame(rafId);
      resizeObserver?.disconnect();
      if (onResize) window.removeEventListener("resize", onResize);
    };
  }, [items, radius, speed, heroRef, anchorRef]);

  return (
    <div
      ref={ref}
      aria-hidden="true"
      className={`${styles.orbit} pointer-events-none absolute left-1/2 z-raised h-0 w-0 ${className}`}
      style={{ animationDelay: delay }}
    >
      {items.map((item) => (
        <div
          key={item.name}
          className="absolute left-0 top-0 flex w-max items-center gap-2 whitespace-nowrap rounded-full bg-[rgba(14,18,32,0.72)] py-1.5 pl-1.5 pr-3 text-[13.5px] text-[rgba(236,240,255,0.9)] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.1)] backdrop-blur-sm transition-opacity duration-300"
        >
          <img src={item.logo.src} alt="" className="size-6 flex-none rounded-full bg-[#f4f5fa] object-contain p-1" />
          <span className="max-sm:hidden">{item.name}</span>
        </div>
      ))}
    </div>
  );
}

function LogoWall({ title, items, footnote }: { title: string; items: LogoItem[]; footnote: string }) {
  return (
    <figure className="m-0 rounded-2xl bg-white/[0.03] p-6 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]">
      <figcaption className="mb-4 text-xs uppercase tracking-[0.16em] text-[#a6aab3]">{title}</figcaption>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-2.5">
        {items.map((item) => (
          <div
            key={item.name}
            className="flex items-center gap-2.5 rounded-xl bg-white/[0.04] px-3 py-2.5 text-[15px] text-[#f5f6fa]"
          >
            <img src={item.logo.src} alt="" className="size-[30px] rounded-lg bg-[#f4f5fa] object-contain p-[5px]" />
            <span>{item.name}</span>
          </div>
        ))}
      </div>
      <p className="mb-0 mt-4 text-sm text-[#a6aab3]">{footnote}</p>
    </figure>
  );
}

function GithubCta({ label = "Star Moyai on GitHub" }: { label?: string }) {
  return (
    <a
      href={MOYAI_GITHUB_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={`${styles.cta} group inline-flex items-center gap-3 rounded-full px-7 py-3.5 text-base font-semibold text-[#05070d] transition-transform hover:scale-[1.03]`}
    >
      <Github className="size-5" />
      <span>{label}</span>
      <Star className="size-4 fill-[#ffb98a] text-[#ffb98a] transition-transform group-hover:rotate-[72deg]" />
    </a>
  );
}

function SecondaryCta({ href, icon, children }: { href: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-2 rounded-full bg-white/[0.06] px-5 py-3 text-sm font-medium text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.18)] backdrop-blur transition-colors hover:bg-white/[0.12]"
    >
      {icon}
      {children}
    </a>
  );
}

function DemoPreview() {
  const [failed, setFailed] = useState(false);
  return (
    <a
      href={MOYAI_WALKTHROUGH_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="group relative block overflow-hidden rounded-2xl bg-[#0b1020] shadow-[0_0_0_1px_rgba(159,171,255,0.22),0_30px_120px_-20px_rgba(120,170,255,0.45)]"
    >
      <div className="flex items-center gap-1.5 border-b border-white/10 bg-white/[0.04] px-4 py-2.5">
        <span className="size-2.5 rounded-full bg-[#ff5f57]" />
        <span className="size-2.5 rounded-full bg-[#febc2e]" />
        <span className="size-2.5 rounded-full bg-[#28c840]" />
        <span className="ml-3 truncate text-xs text-[#a6a4c6]">github.com/BerriAI/moyai</span>
      </div>
      {failed ? (
        <div className="flex aspect-[1196/720] flex-col items-center justify-center gap-3 text-[#c2c6d0]">
          <Play className="size-10" />
          <span>Watch the demo on GitHub</span>
        </div>
      ) : (
        <img
          src={MOYAI_DEMO_GIF_URL}
          alt="Moyai demo: picking a harness and model, then running a task"
          className="block aspect-[1196/720] w-full bg-white object-cover"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      )}
      <span className="absolute bottom-4 right-4 inline-flex items-center gap-1.5 rounded-full bg-[#05070d]/80 px-3 py-1.5 text-xs font-medium text-white opacity-0 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.15)] transition-opacity group-hover:opacity-100">
        Full walkthrough on GitHub <ArrowUpRight className="size-3.5" />
      </span>
    </a>
  );
}

function QuickConnectDialog({ onQuickConnect }: { onQuickConnect: (url: string) => Promise<void> | void }) {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  const connect = async () => {
    const normalized = normalizeMoyaiUrl(url);
    if (!normalized) {
      setError("Enter a full http or https URL, without credentials");
      return;
    }
    setError(null);
    setPending(true);
    try {
      await onQuickConnect(normalized);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start quick connect");
      setPending(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger
        render={
          <button
            type="button"
            className={`${styles.connect} inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-sm font-semibold text-white transition-transform hover:scale-[1.03]`}
          />
        }
      >
        <Plug aria-hidden className="size-4 text-[#cfe3ff]" />
        Quick connect
        <ArrowRight aria-hidden className="size-4" />
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Quick connect Moyai</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-2">
          <label htmlFor="moyai-quick-connect-url" className="text-sm font-medium">
            Moyai URL
          </label>
          <input
            id="moyai-quick-connect-url"
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://moyai.your-company.com"
            className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
          />
          <p className="m-0 text-xs text-muted-foreground">
            You&rsquo;ll confirm on your Moyai workspace as an admin, then come straight back
          </p>
          {error && (
            <p role="alert" className="m-0 text-xs text-red-500">
              {error}
            </p>
          )}
        </div>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            disabled={pending}
            onClick={connect}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {pending ? "Connecting" : "Connect"}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

const STATS = [
  { value: "79%", label: "cheaper than our Devin bill" },
  { value: "100+", label: "providers through LiteLLM" },
  { value: "6", label: "agent harnesses" },
];

export default function MoyaiLanding({
  canQuickConnect = false,
  onQuickConnect,
}: {
  canQuickConnect?: boolean;
  onQuickConnect?: (url: string) => Promise<void> | void;
}) {
  const heroRef = useRef<HTMLElement>(null);
  const anchorRef = useRef<HTMLDivElement>(null);
  const starsRef = useCanvasScene(startStarfield);
  const planetRef = useCanvasScene(startPlanetrise);

  return (
    <div className="relative isolate min-h-full bg-[radial-gradient(ellipse_at_50%_70%,#0a1226_0%,#04060c_65%)] text-[#e3e3e3]">
      <canvas ref={starsRef} aria-hidden="true" className="pointer-events-none absolute inset-0 size-full" />

      <section ref={heroRef} className="relative isolate h-[calc(100vh-3.5rem)] min-h-[680px] overflow-hidden">
        <canvas ref={planetRef} aria-hidden="true" className="pointer-events-none absolute inset-0 size-full" />

        <div ref={anchorRef} className="relative z-sticky flex flex-col items-center px-6 pt-[4vh] text-center">
          <div
            className={`${styles.rise} mb-5 inline-flex items-center gap-2 rounded-full bg-white/[0.06] px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-[rgba(220,228,255,0.75)] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)]`}
          >
            <span className="size-1.5 rounded-full bg-[#28c840] shadow-[0_0_8px_#28c840]" />
            Now open source
          </div>
          <h1
            className={`${styles.rise} m-0 flex flex-col items-center text-[clamp(56px,min(8vw,11vh),112px)] font-medium leading-none tracking-[-0.04em] text-white [text-shadow:0_0_40px_rgba(3,4,8,0.9)]`}
            style={{ animationDelay: "0.08s" }}
          >
            <img src={moyaiHead.src} alt="" className={`${styles.mark} mb-[0.08em] h-[0.78em] w-auto`} />
            <span>Moyai</span>
          </h1>
          <p
            className={`${styles.rise} mb-0 mt-3 text-[clamp(18px,1.8vw,26px)] tracking-[-0.01em] text-[rgba(226,234,255,0.86)]`}
            style={{ animationDelay: "0.16s" }}
          >
            The open source cloud coding agent
          </p>
          <div
            className={`${styles.rise} mt-7 flex flex-wrap items-center justify-center gap-3`}
            style={{ animationDelay: "0.26s" }}
          >
            <GithubCta />
            <SecondaryCta href={MOYAI_WALKTHROUGH_URL} icon={<Play className="size-4" />}>
              Watch the demo
            </SecondaryCta>
          </div>
          <div
            className={`${styles.rise} mt-6 text-[15px] text-[rgba(226,234,255,0.82)]`}
            style={{ animationDelay: "0.34s" }}
          >
            {canQuickConnect && onQuickConnect ? (
              <span className="inline-flex flex-wrap items-center justify-center gap-3 rounded-full bg-white/[0.05] py-1.5 pl-5 pr-1.5 shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)] backdrop-blur">
                Already have Moyai deployed?
                <QuickConnectDialog onQuickConnect={onQuickConnect} />
              </span>
            ) : (
              <span>Already have Moyai deployed? Ask a proxy admin to connect it</span>
            )}
          </div>
        </div>

        <Orbit items={HARNESSES} radius={INNER_RADIUS} speed={0.07} heroRef={heroRef} anchorRef={anchorRef} />
        <Orbit
          items={PROVIDERS}
          radius={OUTER_RADIUS}
          speed={-0.045}
          heroRef={heroRef}
          anchorRef={anchorRef}
          delay="0.3s"
          className="max-md:hidden"
        />

        <div className="absolute inset-x-0 bottom-[6vh] z-chrome flex flex-col items-center gap-1 px-6 text-center [text-shadow:0_0_30px_rgba(3,4,8,0.95)]">
          <span className="text-[clamp(18px,2vw,28px)] font-medium tracking-[-0.02em] text-white">
            Works with Claude Code and Codex
          </span>
          <span className="text-[15px] text-[rgba(240,236,228,0.8)]">Self-hosted · 100+ providers through LiteLLM</span>
        </div>
      </section>

      <div className="relative z-raised mx-auto max-w-5xl px-6 pb-24">
        <div className="grid grid-cols-3 gap-3 pt-14 max-sm:grid-cols-1">
          {STATS.map((s) => (
            <div
              key={s.value}
              className="rounded-2xl bg-white/[0.03] px-6 py-5 text-center shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
            >
              <div className="bg-gradient-to-b from-white to-[#9cc3ff] bg-clip-text text-4xl font-semibold tracking-[-0.03em] text-transparent">
                {s.value}
              </div>
              <div className="mt-1 text-sm text-[#a6aab3]">{s.label}</div>
            </div>
          ))}
        </div>

        <h2 className="mb-3 mt-20 text-center text-[clamp(28px,3.2vw,44px)] font-medium leading-tight tracking-[-0.03em] text-white">
          See it in action
        </h2>
        <p className="mx-auto mb-8 mt-0 max-w-2xl text-center text-[17px] text-[#c2c6d0]">
          Give it a task from Slack or the browser. It edits code, runs your tests in its own cloud workspace, and opens
          a pull request while your laptop is closed.
        </p>
        <DemoPreview />

        <div className="mt-16 grid gap-4 md:grid-cols-2">
          <LogoWall
            title="Any harness"
            items={HARNESSES}
            footnote="Pick the harness per session. Same workspace, tools and permissions."
          />
          <LogoWall
            title="Any model, any provider"
            items={PROVIDERS}
            footnote="Routed through your LiteLLM gateway, with spend tracked per teammate."
          />
        </div>

        <div className="relative mt-20 overflow-hidden rounded-3xl px-8 py-14 text-center shadow-[0_0_0_1px_rgba(159,171,255,0.2)]">
          <div
            aria-hidden="true"
            className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_120%,rgba(120,170,255,0.45)_0%,rgba(40,60,140,0.15)_45%,rgba(5,7,13,0)_75%)]"
          />
          <img src={moyaiHead.src} alt="" className={`${styles.mark} relative z-raised mx-auto mb-4 h-14 w-auto`} />
          <h2 className="relative z-raised m-0 text-[clamp(26px,3vw,40px)] font-medium tracking-[-0.03em] text-white">
            Run your own cloud agent
          </h2>
          <p className="relative z-raised mx-auto mb-7 mt-3 max-w-xl text-[16px] text-[#c2c6d0]">
            Deploy it on your own infrastructure and point it at this gateway. Your code and credentials stay in your
            accounts.
          </p>
          <code className="relative z-raised mx-auto mb-8 block w-fit max-w-full overflow-x-auto rounded-xl bg-black/50 px-5 py-3 text-left font-mono text-sm text-[#cfe3ff] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.1)]">
            git clone {MOYAI_GITHUB_URL}.git
          </code>
          <div className="relative z-raised flex flex-wrap items-center justify-center gap-3">
            <GithubCta label="Get Moyai on GitHub" />
            <SecondaryCta href={MOYAI_LAUNCH_POST_URL} icon={<ArrowUpRight className="size-4" />}>
              Read the launch post
            </SecondaryCta>
          </div>
        </div>
      </div>
    </div>
  );
}
