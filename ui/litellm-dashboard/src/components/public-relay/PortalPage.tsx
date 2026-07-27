"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { ApiKey, Money, relayFetch } from "@/lib/http/publicRelay";
import { usePortal } from "./PortalContext";
import { migratedHref } from "@/utils/migratedPages";

type Section = "overview" | "keys" | "credit" | "usage" | "logs" | "security";
type Wallet = { available: Money; reserved: Money };
type Usage = {
  request_count: number;
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  charged: Money;
  upstream_cost: Money;
};
type LedgerEntry = { entry_id: string; entry_type: string; amount: Money; created_at: string };
type Price = {
  price_id: string;
  model_name: string;
  input_micros_per_million: number;
  output_micros_per_million: number | null;
};
type RequestLog = {
  request_id: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  charged: Money;
  status: string | null;
  request_duration_ms: number | null;
  created_at: string;
};

const titles: Record<Section, [string, string]> = {
  overview: ["Overview", "Your balance and relay activity at a glance."],
  keys: ["API keys", "Create up to five scoped keys. Full values appear only once."],
  credit: ["额度与账目", "企业额度由管理员线下授予，永久有效。"],
  usage: ["Usage", "Token consumption and customer charges across your account."],
  logs: ["Request logs", "Seven days of request metadata without plaintext prompts."],
  security: ["Account security", "Manage the portal session connected to your verified email."],
};

export default function PortalPage({ section }: { section: Section }) {
  const [title, subtitle] = titles[section];
  return (
    <div className="mx-auto max-w-6xl">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-[#d83a20]">User portal</p>
      <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em]">{title}</h1>
      <p className="mt-3 text-black/55">{subtitle}</p>
      <div className="mt-9">
        {section === "overview" && <Overview />}
        {section === "keys" && <Keys />}
        {section === "credit" && <Credit />}
        {section === "usage" && <UsageView />}
        {section === "logs" && <Logs />}
        {section === "security" && <Security />}
      </div>
    </div>
  );
}

function Overview() {
  const [wallet, setWallet] = useState<Wallet | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);

  useEffect(() => {
    Promise.all([relayFetch<Wallet>("/v1/portal/wallet"), relayFetch<Usage>("/v1/portal/usage")]).then(
      ([walletResult, usageResult]) => {
        setWallet(walletResult.data);
        setUsage(usageResult.data);
      },
    );
  }, []);

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Metric label="Available balance" value={wallet?.available.display ?? "—"} accent />
      <Metric label="Reserved" value={wallet?.reserved.display ?? "—"} />
      <Metric label="Requests" value={usage?.request_count.toLocaleString() ?? "—"} />
      <div className="rounded-3xl border border-black/10 bg-white p-6 md:col-span-3">
        <p className="text-sm text-black/45">Getting started</p>
        <p className="mt-3 max-w-2xl text-lg leading-8">
          联系管理员授予额度，创建服务端 API Key，然后从{" "}
          <code className="rounded bg-black/5 px-1.5 py-1 text-sm">GET /v1/models</code>.
        </p>
      </div>
    </div>
  );
}

function Keys() {
  const { csrfToken } = usePortal();
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [alias, setAlias] = useState("");
  const [logContent, setLogContent] = useState(false);
  const [newKey, setNewKey] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => {
    relayFetch<{ data: ApiKey[] }>("/v1/portal/keys")
      .then(({ data }) => setKeys(data.data))
      .catch((requestError: Error) => setError(requestError.message));
  }, []);

  useEffect(load, [load]);

  async function create(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const { data } = await relayFetch<ApiKey>(
        "/v1/portal/keys",
        { method: "POST", body: JSON.stringify({ alias, log_content: logContent }) },
        csrfToken,
      );
      setNewKey(data.key ?? "");
      setAlias("");
      load();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to create key");
    }
  }

  async function remove(keyId: string) {
    await relayFetch(`/v1/portal/keys/${encodeURIComponent(keyId)}`, { method: "DELETE" }, csrfToken);
    load();
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[.8fr_1.2fr]">
      <form onSubmit={create} className="rounded-3xl border border-black/10 bg-white p-6">
        <h2 className="text-lg font-semibold">Create key</h2>
        <label className="mt-5 block text-sm">
          Alias
          <input
            required
            value={alias}
            onChange={(event) => setAlias(event.target.value)}
            className="mt-2 w-full rounded-xl border-black/15"
          />
        </label>
        <label className="mt-4 flex items-start gap-3 text-sm text-black/65">
          <input
            type="checkbox"
            checked={logContent}
            onChange={(event) => setLogContent(event.target.checked)}
            className="mt-1 rounded"
          />
          Encrypt request content for seven-day logs
        </label>
        <button className="mt-6 rounded-full bg-[#11110f] px-5 py-2.5 text-sm text-white">Create key</button>
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
      </form>
      <div className="space-y-3">
        {newKey && (
          <div className="rounded-3xl bg-[#d9e8ff] p-6">
            <p className="text-sm font-medium">Copy this key now</p>
            <code className="mt-3 block overflow-x-auto text-sm">{newKey}</code>
            <button
              type="button"
              onClick={() => navigator.clipboard.writeText(newKey)}
              className="mt-4 rounded-full bg-black px-4 py-2 text-xs text-white"
            >
              Copy
            </button>
          </div>
        )}
        {keys.map((key) => (
          <div
            key={key.key_id}
            className="flex items-center justify-between rounded-2xl border border-black/10 bg-white p-5"
          >
            <div>
              <p className="font-medium">{key.alias || "Untitled key"}</p>
              <p className="mt-1 font-mono text-xs text-black/40">{key.key_id.slice(0, 16)}…</p>
            </div>
            <button type="button" onClick={() => remove(key.key_id)} className="text-sm text-red-700">
              Delete
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

function Credit() {
  const [wallet, setWallet] = useState<Wallet | null>(null);
  const [ledger, setLedger] = useState<LedgerEntry[]>([]);
  const [prices, setPrices] = useState<Price[]>([]);
  useEffect(() => {
    Promise.all([
      relayFetch<Wallet>("/v1/portal/wallet"),
      relayFetch<{ data: LedgerEntry[] }>("/v1/portal/ledger"),
      relayFetch<{ models: Price[] }>("/v1/portal/pricing"),
    ]).then(([walletResult, ledgerResult, priceResult]) => {
      setWallet(walletResult.data);
      setLedger(ledgerResult.data.data);
      setPrices(priceResult.data.models);
    });
  }, []);
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="rounded-3xl border border-black/10 bg-white p-6">
        <h2 className="text-lg font-semibold">当前额度</h2>
        <p className="mt-5 text-4xl font-semibold">{wallet?.available.display ?? "—"}</p>
        <p className="mt-2 text-sm text-black/45">预留 {wallet?.reserved.display ?? "—"}</p>
      </div>
      <div className="rounded-3xl border border-black/10 bg-white p-6">
        <h2 className="text-lg font-semibold">统一价格</h2>
        <div className="mt-5 divide-y divide-black/10">
          {prices.map((price) => (
            <div key={price.price_id} className="flex justify-between py-4 text-sm">
              <span>{price.model_name}</span>
              <span className="font-mono text-xs text-black/55">
                {price.input_micros_per_million} / {price.output_micros_per_million ?? "—"} μUSD
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="rounded-3xl border border-black/10 bg-white p-6 lg:col-span-2">
        <h2 className="text-lg font-semibold">额度账目</h2>
        <div className="mt-5 divide-y divide-black/10">
          {ledger.map((entry) => (
            <div key={entry.entry_id} className="flex justify-between py-4 text-sm">
              <span>{entry.entry_type}</span>
              <span>{entry.amount.display}</span>
            </div>
          ))}
          {!ledger.length && <p className="py-8 text-sm text-black/45">暂无账目。</p>}
        </div>
      </div>
    </div>
  );
}

function UsageView() {
  const [usage, setUsage] = useState<Usage | null>(null);
  useEffect(() => {
    relayFetch<Usage>("/v1/portal/usage").then(({ data }) => setUsage(data));
  }, []);
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Metric label="Customer charge" value={usage?.charged.display ?? "—"} accent />
      <Metric label="Input tokens" value={usage?.input_tokens.toLocaleString() ?? "—"} />
      <Metric label="Output tokens" value={usage?.output_tokens.toLocaleString() ?? "—"} />
      <Metric label="Cached input" value={usage?.cached_input_tokens.toLocaleString() ?? "—"} />
      <Metric label="Requests" value={usage?.request_count.toLocaleString() ?? "—"} />
    </div>
  );
}

function Logs() {
  const [logs, setLogs] = useState<RequestLog[]>([]);
  useEffect(() => {
    relayFetch<{ data: RequestLog[] }>("/v1/portal/logs").then(({ data }) => setLogs(data.data));
  }, []);
  return (
    <div className="overflow-x-auto rounded-3xl border border-black/10 bg-white">
      <table className="w-full min-w-[760px] text-left text-sm">
        <thead className="border-b border-black/10 text-black/45">
          <tr>
            <th className="p-5">Model</th>
            <th className="p-5">Tokens</th>
            <th className="p-5">Charge</th>
            <th className="p-5">Status</th>
            <th className="p-5">Latency</th>
          </tr>
        </thead>
        <tbody>
          {logs.map((log) => (
            <tr key={log.request_id} className="border-b border-black/5">
              <td className="p-5">
                <p>{log.model}</p>
                <p className="mt-1 font-mono text-xs text-black/35">{log.request_id.slice(0, 12)}</p>
              </td>
              <td className="p-5">{(log.input_tokens + log.output_tokens).toLocaleString()}</td>
              <td className="p-5">{log.charged.display}</td>
              <td className="p-5">{log.status || "complete"}</td>
              <td className="p-5">{log.request_duration_ms === null ? "—" : `${log.request_duration_ms} ms`}</td>
            </tr>
          ))}
          {!logs.length && (
            <tr>
              <td colSpan={5} className="p-8 text-center text-black/45">
                No settled requests yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function Security() {
  const { account, csrfToken } = usePortal();
  const router = useRouter();
  async function logout() {
    await relayFetch("/v1/public/auth/logout", { method: "POST" }, csrfToken);
    router.push(migratedHref("relay-login"));
  }
  return (
    <section className="max-w-2xl rounded-3xl border border-black/10 bg-white p-7">
      <p className="text-sm text-black/45">Verified email</p>
      <p className="mt-2 text-lg font-medium">{account.email}</p>
      <p className="mt-7 text-sm text-black/45">Account status</p>
      <p className="mt-2 text-lg font-medium">{account.status.toLowerCase()}</p>
      <button
        type="button"
        onClick={logout}
        className="mt-8 rounded-full border border-red-200 px-5 py-2.5 text-sm text-red-700"
      >
        Sign out everywhere
      </button>
    </section>
  );
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`rounded-3xl border border-black/10 p-6 ${accent ? "bg-[#d9e8ff]" : "bg-white"}`}>
      <p className="text-sm text-black/45">{label}</p>
      <p className="mt-5 text-3xl font-semibold tracking-tight">{value}</p>
    </div>
  );
}
