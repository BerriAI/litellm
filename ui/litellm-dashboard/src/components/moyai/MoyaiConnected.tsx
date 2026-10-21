"use client";

import React from "react";
import { ArrowUpRight, Check, ExternalLink } from "lucide-react";
import moyaiHead from "../../../public/assets/moyai/moyai-head.svg";
import styles from "./MoyaiLanding.module.css";
import { uiHref } from "@/utils/uiHref";

export default function MoyaiConnected({
  moyaiUrl,
  keyAlias,
  models,
}: {
  moyaiUrl: string | null;
  keyAlias?: string | null;
  models?: number | null;
}) {
  const rows = [
    { label: "Workspace linked", value: moyaiUrl },
    { label: "Virtual key issued", value: keyAlias },
    { label: "models available through LiteLLM", value: models != null ? `${models}` : null, prefix: true },
  ];

  return (
    <div className="relative isolate min-h-full bg-[radial-gradient(ellipse_at_50%_70%,#0a1226_0%,#04060c_65%)] text-[#e3e3e3]">
      <div className="mx-auto flex min-h-[calc(100vh-3.5rem)] max-w-2xl flex-col items-center justify-center px-6 py-16 text-center">
        <div
          className={`${styles.rise} mb-5 inline-flex items-center gap-2 rounded-full bg-white/[0.06] px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-[rgba(220,228,255,0.75)] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.12)]`}
        >
          <span className="size-1.5 rounded-full bg-[#28c840] shadow-[0_0_8px_#28c840]" />
          Connected
        </div>
        <img src={moyaiHead.src} alt="" className={`${styles.mark} mb-4 h-14 w-auto`} />
        <h1 className="m-0 text-[clamp(28px,3.5vw,44px)] font-medium tracking-[-0.03em] text-white">Moyai connected</h1>
        <p className="mb-0 mt-3 max-w-md text-[16px] text-[#c2c6d0]">
          This deployment is now linked to your LiteLLM gateway.
        </p>

        <div className="mt-8 w-full max-w-md space-y-2 text-left">
          {rows.map((row) =>
            row.value ? (
              <div
                key={row.label}
                className="flex items-center gap-3 rounded-xl bg-white/[0.04] px-4 py-3 text-[15px] text-[#f5f6fa] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.08)]"
              >
                <Check className="size-4 flex-none text-[#28c840]" />
                <span>
                  {row.prefix ? (
                    <>
                      <strong className="font-semibold">{row.value}</strong> {row.label}
                    </>
                  ) : (
                    <>
                      {row.label} <span className="text-[#a6aab3]">{row.value}</span>
                    </>
                  )}
                </span>
              </div>
            ) : null,
          )}
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          {moyaiUrl && (
            <a
              href={moyaiUrl}
              className={`${styles.cta} group inline-flex items-center gap-3 rounded-full px-7 py-3.5 text-base font-semibold text-[#05070d] transition-transform hover:scale-[1.03]`}
            >
              Open Moyai
              <ArrowUpRight className="size-4" />
            </a>
          )}
          <a
            href={uiHref("")}
            className="inline-flex items-center gap-2 rounded-full bg-white/[0.06] px-5 py-3 text-sm font-medium text-white shadow-[inset_0_0_0_1px_rgba(255,255,255,0.18)] backdrop-blur transition-colors hover:bg-white/[0.12]"
          >
            <ExternalLink className="size-4" />
            Back to AI Gateway
          </a>
        </div>
      </div>
    </div>
  );
}
