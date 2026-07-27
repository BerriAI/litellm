"use client";

import Link from "next/link";
import { migratedHref } from "@/utils/migratedPages";

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-[#f7f7f2] text-[#11110f]">
      <header className="sticky top-0 z-20 border-b border-black/10 bg-[#f7f7f2]/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 lg:px-8">
          <Link href={migratedHref("")} className="flex items-center gap-3 font-semibold tracking-tight">
            <span className="grid size-8 place-items-center rounded-full bg-[#ff4f2e] text-sm text-white">L</span>
            LiteLLM 企业中转站
          </Link>
          <nav className="hidden items-center gap-7 text-sm md:flex">
            <Link href={migratedHref("docs")}>Docs</Link>
            <Link href={migratedHref("status")}>Status</Link>
          </nav>
          <div className="flex items-center gap-3 text-sm">
            <Link href={migratedHref("relay-login")} className="px-3 py-2">
              Sign in
            </Link>
          </div>
        </div>
      </header>
      {children}
      <footer className="border-t border-black/10 px-5 py-10 text-sm text-black/60">
        <div className="mx-auto flex max-w-7xl flex-col justify-between gap-4 sm:flex-row">
          <p>面向企业的 OpenAI 兼容模型访问，由 LiteLLM 驱动。</p>
          <div className="flex gap-5">
            <Link href={migratedHref("docs")}>API docs</Link>
            <Link href={migratedHref("status")}>Service status</Link>
            <Link href={migratedHref("admin")}>Admin</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
