"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

import { useAuth } from "@/contexts/AuthContext";
import { Money, relayFetch } from "@/lib/http/publicRelay";

type Account = {
  account_id: string;
  email: string;
  status: string;
  wallet_id: string;
  available: Money;
  debt: Money;
};
type Payment = {
  payment_id: string;
  email: string;
  amount: Money;
  refunded: Money;
  status: string;
};
type Price = {
  price_id: string;
  model_name: string;
  version: number;
  input_micros_per_million: number;
  output_micros_per_million: number | null;
  enabled: boolean;
};

export default function RelayAdmin() {
  const { accessToken } = useAuth();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [payments, setPayments] = useState<Payment[]>([]);
  const [prices, setPrices] = useState<Price[]>([]);
  const [margin, setMargin] = useState<{ charged: Money; upstream_cost: Money; gross_margin: Money } | null>(null);
  const [error, setError] = useState("");
  const [model, setModel] = useState("");
  const [inputRate, setInputRate] = useState(1_000_000);
  const [outputRate, setOutputRate] = useState(4_000_000);
  const [cachedInputRate, setCachedInputRate] = useState("");
  const [embeddingRate, setEmbeddingRate] = useState("");
  const [defaultOutput, setDefaultOutput] = useState(4096);
  const [maximumOutput, setMaximumOutput] = useState(8192);

  const request = useCallback(
    async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
      const { data } = await relayFetch<T>(path, {
        ...init,
        headers: { authorization: `Bearer ${accessToken}`, ...init.headers },
      });
      return data;
    },
    [accessToken],
  );

  const load = useCallback(async () => {
    if (!accessToken) return;
    try {
      const [accountResult, paymentResult, priceResult, marginResult] = await Promise.all([
        request<{ data: Account[] }>("/v1/admin/relay/accounts"),
        request<{ data: Payment[] }>("/v1/admin/relay/payments"),
        request<{ models: Price[] }>("/v1/admin/relay/prices"),
        request<{ charged: Money; upstream_cost: Money; gross_margin: Money }>("/v1/admin/relay/margin"),
      ]);
      setAccounts(accountResult.data);
      setPayments(paymentResult.data);
      setPrices(priceResult.models);
      setMargin(marginResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to load relay administration");
    }
  }, [accessToken, request]);

  useEffect(() => {
    const timeout = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timeout);
  }, [load]);

  async function publish(event: FormEvent) {
    event.preventDefault();
    await request("/v1/admin/relay/prices", {
      method: "POST",
      body: JSON.stringify({
        model_name: model,
        input_micros_per_million: inputRate,
        cached_input_micros_per_million: cachedInputRate === "" ? null : Number(cachedInputRate),
        output_micros_per_million: outputRate,
        embedding_micros_per_million: embeddingRate === "" ? null : Number(embeddingRate),
        default_max_output_tokens: defaultOutput,
        max_output_tokens: maximumOutput,
        enabled: true,
      }),
    });
    setModel("");
    load();
  }

  async function toggleAccount(account: Account) {
    await request(`/v1/admin/relay/accounts/${account.account_id}/status`, {
      method: "POST",
      body: JSON.stringify({ status: account.status === "ACTIVE" ? "FROZEN" : "ACTIVE" }),
    });
    load();
  }

  async function adjustAccount(account: Account) {
    const amount = window.prompt("Wallet adjustment in USD. Use a negative value to debit.", "5");
    if (!amount) return;
    const amountMicros = Math.round(Number(amount) * 1_000_000);
    if (!Number.isSafeInteger(amountMicros) || amountMicros === 0) return;
    await request(`/v1/admin/relay/wallets/${account.wallet_id}/adjust`, {
      method: "POST",
      headers: { "idempotency-key": crypto.randomUUID() },
      body: JSON.stringify({ amount_micros: amountMicros, reason: "Administrator adjustment" }),
    });
    load();
  }

  async function refund(payment: Payment) {
    const amount = window.prompt("Refund amount in USD", "5");
    if (!amount) return;
    const amountMicros = Math.round(Number(amount) * 1_000_000);
    await request(`/v1/admin/relay/payments/${payment.payment_id}/refund`, {
      method: "POST",
      headers: { "idempotency-key": crypto.randomUUID() },
      body: JSON.stringify({ amount_micros: amountMicros, reason: "Administrator refund" }),
    });
    load();
  }

  return (
    <div className="mx-auto max-w-7xl p-6">
      <div className="flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
        <div>
          <p className="text-sm text-gray-500">Public relay</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">Operations</h1>
        </div>
        <div className="grid grid-cols-3 gap-3 text-right text-sm">
          <Summary label="Revenue" value={margin?.charged.display ?? "—"} />
          <Summary label="Upstream" value={margin?.upstream_cost.display ?? "—"} />
          <Summary label="Gross margin" value={margin?.gross_margin.display ?? "—"} />
        </div>
      </div>
      {error && <p className="mt-6 rounded-xl bg-red-50 p-4 text-sm text-red-700">{error}</p>}
      <div className="mt-8 grid gap-6 xl:grid-cols-[.7fr_1.3fr]">
        <form onSubmit={publish} className="rounded-2xl border border-gray-200 bg-white p-6">
          <h2 className="text-lg font-semibold">Publish price version</h2>
          <label className="mt-5 block text-sm">
            Model
            <input
              required
              value={model}
              onChange={(event) => setModel(event.target.value)}
              className="mt-2 w-full rounded-lg border-gray-200"
            />
          </label>
          <label className="mt-4 block text-sm">
            Input micros / 1M
            <input
              type="number"
              min={0}
              value={inputRate}
              onChange={(event) => setInputRate(Number(event.target.value))}
              className="mt-2 w-full rounded-lg border-gray-200"
            />
          </label>
          <label className="mt-4 block text-sm">
            Cached input micros / 1M
            <input
              type="number"
              min={0}
              placeholder="Use input rate"
              value={cachedInputRate}
              onChange={(event) => setCachedInputRate(event.target.value)}
              className="mt-2 w-full rounded-lg border-gray-200"
            />
          </label>
          <label className="mt-4 block text-sm">
            Output micros / 1M
            <input
              type="number"
              min={0}
              value={outputRate}
              onChange={(event) => setOutputRate(Number(event.target.value))}
              className="mt-2 w-full rounded-lg border-gray-200"
            />
          </label>
          <label className="mt-4 block text-sm">
            Embedding micros / 1M
            <input
              type="number"
              min={0}
              placeholder="Not published"
              value={embeddingRate}
              onChange={(event) => setEmbeddingRate(event.target.value)}
              className="mt-2 w-full rounded-lg border-gray-200"
            />
          </label>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <label className="block text-sm">
              Default output
              <input
                type="number"
                min={1}
                value={defaultOutput}
                onChange={(event) => setDefaultOutput(Number(event.target.value))}
                className="mt-2 w-full rounded-lg border-gray-200"
              />
            </label>
            <label className="block text-sm">
              Maximum output
              <input
                type="number"
                min={1}
                value={maximumOutput}
                onChange={(event) => setMaximumOutput(Number(event.target.value))}
                className="mt-2 w-full rounded-lg border-gray-200"
              />
            </label>
          </div>
          <button className="mt-5 rounded-lg bg-black px-4 py-2 text-sm text-white">Publish immutable version</button>
        </form>
        <Panel title="Active public prices">
          {prices.map((price) => (
            <Row
              key={price.price_id}
              primary={price.model_name}
              secondary={`v${price.version}`}
              value={`${price.input_micros_per_million} / ${price.output_micros_per_million ?? "—"}`}
            />
          ))}
        </Panel>
      </div>
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Panel title="Public accounts">
          {accounts.map((account) => (
            <div
              key={account.account_id}
              className="flex items-center justify-between border-b border-gray-100 py-4 last:border-0"
            >
              <div>
                <p className="text-sm font-medium">{account.email}</p>
                <p className="mt-1 text-xs text-gray-400">
                  {account.available.display} · {account.status}
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => adjustAccount(account)}
                  className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs"
                >
                  Adjust
                </button>
                <button
                  onClick={() => toggleAccount(account)}
                  className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs"
                >
                  {account.status === "ACTIVE" ? "Freeze" : "Activate"}
                </button>
              </div>
            </div>
          ))}
        </Panel>
        <Panel title="Payment orders">
          {payments.map((payment) => (
            <div
              key={payment.payment_id}
              className="flex items-center justify-between border-b border-gray-100 py-4 last:border-0"
            >
              <div>
                <p className="text-sm font-medium">{payment.email}</p>
                <p className="mt-1 text-xs text-gray-400">
                  {payment.amount.display} · {payment.status}
                </p>
              </div>
              {["PAID", "PARTIALLY_REFUNDED"].includes(payment.status) && (
                <button
                  onClick={() => refund(payment)}
                  className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs"
                >
                  Refund
                </button>
              )}
            </div>
          ))}
        </Panel>
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function Row({ primary, secondary, value }: { primary: string; secondary: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-gray-100 py-4 last:border-0">
      <div>
        <p className="text-sm font-medium">{primary}</p>
        <p className="mt-1 text-xs text-gray-400">{secondary}</p>
      </div>
      <span className="font-mono text-xs text-gray-500">{value}</span>
    </div>
  );
}

function Summary({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-400">{label}</p>
      <p className="mt-1 font-semibold">{value}</p>
    </div>
  );
}
