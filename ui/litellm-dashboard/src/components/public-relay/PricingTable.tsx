"use client";

import { useEffect, useState } from "react";

import { relayFetch } from "@/lib/http/publicRelay";

type Price = {
  price_id: string;
  model_name: string;
  version: number;
  input_micros_per_million: number;
  cached_input_micros_per_million: number | null;
  output_micros_per_million: number | null;
  embedding_micros_per_million: number | null;
  max_output_tokens: number;
};

export default function PricingTable() {
  const [prices, setPrices] = useState<Price[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    relayFetch<{ models: Price[] }>("/v1/public/pricing")
      .then(({ data }) => setPrices(data.models))
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  if (error) {
    return <p className="mt-10 rounded-2xl bg-red-50 p-5 text-red-800">{error}</p>;
  }

  return (
    <div className="mt-12 overflow-x-auto rounded-3xl border border-black/10 bg-white">
      <table className="w-full min-w-[760px] text-left text-sm">
        <thead className="border-b border-black/10 text-black/45">
          <tr>
            <th className="p-5 font-medium">Model</th>
            <th className="p-5 font-medium">Input</th>
            <th className="p-5 font-medium">Cached input</th>
            <th className="p-5 font-medium">Output</th>
            <th className="p-5 font-medium">Embedding</th>
          </tr>
        </thead>
        <tbody>
          {prices.map((price) => (
            <tr key={price.price_id} className="border-b border-black/5 last:border-0">
              <td className="p-5">
                <p className="font-medium">{price.model_name}</p>
                <p className="mt-1 text-xs text-black/40">Price v{price.version}</p>
              </td>
              <td className="p-5">{rate(price.input_micros_per_million)}</td>
              <td className="p-5">{rate(price.cached_input_micros_per_million)}</td>
              <td className="p-5">{rate(price.output_micros_per_million)}</td>
              <td className="p-5">{rate(price.embedding_micros_per_million)}</td>
            </tr>
          ))}
          {!prices.length && (
            <tr>
              <td colSpan={5} className="p-8 text-center text-black/45">
                No public models are published yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function rate(value: number | null): string {
  return value === null ? "—" : `$${(value / 1_000_000).toFixed(4).replace(/0+$/, "").replace(/\.$/, "")}`;
}
