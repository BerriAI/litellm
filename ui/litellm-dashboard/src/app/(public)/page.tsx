"use client";

import Link from "next/link";
import { migratedHref } from "@/utils/migratedPages";

const features = [
  ["One API", "Use familiar OpenAI SDKs for chat, responses, and embeddings."],
  ["Clear pricing", "Every request is charged against an immutable price snapshot."],
  ["Prepaid control", "No surprise invoices. Requests stop when your balance reaches zero."],
];

export default function RelayHome() {
  return (
    <main>
      <section className="overflow-hidden border-b border-black/10">
        <div className="mx-auto grid max-w-7xl gap-16 px-5 py-20 lg:grid-cols-[1.15fr_.85fr] lg:px-8 lg:py-28">
          <div>
            <p className="mb-6 font-mono text-xs uppercase tracking-[0.24em] text-[#d83a20]">The public AI gateway</p>
            <h1 className="max-w-4xl text-5xl font-semibold leading-[0.96] tracking-[-0.055em] sm:text-7xl">
              Ship with the models you need. Keep one clean API.
            </h1>
            <p className="mt-8 max-w-2xl text-lg leading-8 text-black/65">
              A prepaid, OpenAI-compatible relay built on LiteLLM. Create a key, add balance, and move between published
              models without rebuilding your stack.
            </p>
            <div className="mt-10 flex flex-wrap gap-3">
              <Link
                href={migratedHref("register")}
                className="rounded-full bg-[#ff4f2e] px-6 py-3 font-medium text-white"
              >
                Create free account
              </Link>
              <Link href={migratedHref("docs")} className="rounded-full border border-black/20 px-6 py-3 font-medium">
                Read the quickstart
              </Link>
            </div>
          </div>
          <div className="relative min-h-96">
            <div className="absolute inset-0 rotate-3 rounded-[2.5rem] bg-[#d9e8ff]" />
            <div className="absolute inset-4 -rotate-2 rounded-[2.2rem] border border-black/10 bg-[#171713] p-7 text-white shadow-2xl">
              <div className="mb-12 flex items-center justify-between text-xs text-white/50">
                <span>POST /v1/responses</span>
                <span className="text-[#75e7a1]">200 OK</span>
              </div>
              <pre className="overflow-hidden text-sm leading-7 text-white/80">
                <code>{`curl https://api.example.com/v1/responses \\
  -H "Authorization: Bearer $RELAY_KEY" \\
  -d '{
    "model": "published-model",
    "input": "Build something useful"
  }'`}</code>
              </pre>
              <div className="absolute inset-x-7 bottom-7 flex justify-between border-t border-white/10 pt-5 text-xs">
                <span className="text-white/45">Charged from actual tokens</span>
                <span>$0.00284</span>
              </div>
            </div>
          </div>
        </div>
      </section>
      <section className="mx-auto max-w-7xl px-5 py-20 lg:px-8">
        <div className="grid gap-px overflow-hidden rounded-3xl border border-black/10 bg-black/10 md:grid-cols-3">
          {features.map(([title, body], index) => (
            <article key={title} className="bg-[#f7f7f2] p-8">
              <span className="font-mono text-xs text-black/35">0{index + 1}</span>
              <h2 className="mt-12 text-2xl font-semibold tracking-tight">{title}</h2>
              <p className="mt-3 leading-7 text-black/60">{body}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
