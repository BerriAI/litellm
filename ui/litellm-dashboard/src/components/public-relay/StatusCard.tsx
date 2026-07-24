"use client";

import { useEffect, useState } from "react";

import { relayFetch } from "@/lib/http/publicRelay";

export default function StatusCard() {
  const [status, setStatus] = useState<{ enabled: boolean; operational: boolean } | null>(null);

  useEffect(() => {
    relayFetch<{ enabled: boolean; operational: boolean }>("/v1/public/status")
      .then(({ data }) => setStatus(data))
      .catch(() => setStatus({ enabled: false, operational: false }));
  }, []);

  const operational = status?.operational === true;
  let statusLabel = "Checking…";
  if (status !== null) {
    statusLabel = operational ? "All systems operational" : "Relay unavailable";
  }
  return (
    <section className="mt-12 rounded-3xl border border-black/10 bg-white p-8">
      <div className="flex items-center gap-3">
        <span className={`size-3 rounded-full ${operational ? "bg-emerald-500" : "bg-amber-500"}`} />
        <h2 className="text-2xl font-semibold">{statusLabel}</h2>
      </div>
      <p className="mt-4 text-black/55">
        {operational
          ? "Registration, wallet settlement, and published model access are available."
          : "The public relay is disabled or is missing required runtime configuration."}
      </p>
    </section>
  );
}
