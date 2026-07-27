"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";

import { useAuth } from "@/contexts/AuthContext";
import { Money, relayFetch } from "@/lib/http/publicRelay";

type Account = {
  account_id: string;
  company_name: string;
  email: string;
  status: string;
  wallet_id: string;
  available: Money;
};

type Price = {
  price_id: string;
  model_name: string;
  version: number;
  input_micros_per_million: number;
  output_micros_per_million: number | null;
};

type LinkResult = { url: string; expires_at: string };

export default function RelayAdmin() {
  const { accessToken } = useAuth();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [prices, setPrices] = useState<Price[]>([]);
  const [margin, setMargin] = useState<{ charged: Money; upstream_cost: Money; gross_margin: Money } | null>(null);
  const [companyName, setCompanyName] = useState("");
  const [adminEmail, setAdminEmail] = useState("");
  const [initialCredit, setInitialCredit] = useState(0);
  const [notes, setNotes] = useState("");
  const [issuedLink, setIssuedLink] = useState("");
  const [model, setModel] = useState("");
  const [inputRate, setInputRate] = useState(1_000_000);
  const [outputRate, setOutputRate] = useState(4_000_000);
  const [error, setError] = useState("");

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
      const [accountResult, priceResult, marginResult] = await Promise.all([
        request<{ data: Account[] }>("/v1/admin/relay/accounts"),
        request<{ models: Price[] }>("/v1/admin/relay/prices"),
        request<{ charged: Money; upstream_cost: Money; gross_margin: Money }>("/v1/admin/relay/margin"),
      ]);
      setAccounts(accountResult.data);
      setPrices(priceResult.models);
      setMargin(marginResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "无法加载企业中转站管理数据");
    }
  }, [accessToken, request]);

  useEffect(() => {
    const timeout = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timeout);
  }, [load]);

  async function createEnterprise(event: FormEvent) {
    event.preventDefault();
    const result = await request<{ activation: LinkResult }>("/v1/admin/relay/accounts", {
      method: "POST",
      headers: { "idempotency-key": crypto.randomUUID() },
      body: JSON.stringify({
        company_name: companyName,
        admin_email: adminEmail,
        initial_credit_micros: Math.round(initialCredit * 1_000_000),
        notes: notes || null,
      }),
    });
    setIssuedLink(result.activation.url);
    setCompanyName("");
    setAdminEmail("");
    setInitialCredit(0);
    setNotes("");
    void load();
  }

  async function publish(event: FormEvent) {
    event.preventDefault();
    await request("/v1/admin/relay/prices", {
      method: "POST",
      body: JSON.stringify({
        model_name: model,
        input_micros_per_million: inputRate,
        cached_input_micros_per_million: null,
        output_micros_per_million: outputRate,
        embedding_micros_per_million: null,
        default_max_output_tokens: 4096,
        max_output_tokens: 8192,
        enabled: true,
      }),
    });
    setModel("");
    void load();
  }

  async function issueLink(account: Account, kind: "activation-link" | "password-reset-link") {
    const result = await request<LinkResult>(`/v1/admin/relay/accounts/${account.account_id}/${kind}`, {
      method: "POST",
    });
    setIssuedLink(result.url);
  }

  async function adjust(account: Account) {
    const amount = window.prompt("额度调整（USD，扣减请输入负数）", "100");
    const reason = window.prompt("调整原因", "企业线下授信");
    if (!amount || !reason) return;
    const amountMicros = Math.round(Number(amount) * 1_000_000);
    if (!Number.isSafeInteger(amountMicros) || amountMicros === 0) return;
    await request(`/v1/admin/relay/wallets/${account.wallet_id}/adjust`, {
      method: "POST",
      headers: { "idempotency-key": crypto.randomUUID() },
      body: JSON.stringify({ amount_micros: amountMicros, reason }),
    });
    void load();
  }

  async function toggle(account: Account) {
    await request(`/v1/admin/relay/accounts/${account.account_id}/status`, {
      method: "POST",
      body: JSON.stringify({ status: account.status === "ACTIVE" ? "FROZEN" : "ACTIVE" }),
    });
    void load();
  }

  return (
    <div className="mx-auto max-w-7xl p-6">
      <div className="flex flex-wrap items-end justify-between gap-5">
        <div>
          <p className="text-sm text-gray-500">企业中转站</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight">运营管理</h1>
        </div>
        <div className="flex gap-6 text-right text-sm">
          <Summary label="企业收费" value={margin?.charged.display ?? "—"} />
          <Summary label="上游成本" value={margin?.upstream_cost.display ?? "—"} />
          <Summary label="毛利" value={margin?.gross_margin.display ?? "—"} />
        </div>
      </div>
      {error && <p className="mt-6 rounded-xl bg-red-50 p-4 text-sm text-red-700">{error}</p>}
      {issuedLink && (
        <div className="mt-6 rounded-2xl bg-blue-50 p-5">
          <p className="text-sm font-medium">一次性链接（请安全发送给企业管理员）</p>
          <code className="mt-3 block break-all text-xs">{issuedLink}</code>
          <button
            onClick={() => navigator.clipboard.writeText(issuedLink)}
            className="mt-3 rounded-lg bg-black px-3 py-2 text-xs text-white"
          >
            复制链接
          </button>
        </div>
      )}
      <div className="mt-8 grid gap-6 xl:grid-cols-2">
        <Panel title="创建企业账户">
          <form onSubmit={createEnterprise} className="grid gap-4">
            <input
              required
              placeholder="企业名称"
              value={companyName}
              onChange={(event) => setCompanyName(event.target.value)}
              className="rounded-lg border-gray-200"
            />
            <input
              required
              type="email"
              placeholder="管理员邮箱"
              value={adminEmail}
              onChange={(event) => setAdminEmail(event.target.value)}
              className="rounded-lg border-gray-200"
            />
            <input
              type="number"
              min={0}
              step="0.01"
              placeholder="初始额度 USD"
              value={initialCredit}
              onChange={(event) => setInitialCredit(Number(event.target.value))}
              className="rounded-lg border-gray-200"
            />
            <textarea
              placeholder="备注（可选）"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              className="rounded-lg border-gray-200"
            />
            <button className="rounded-lg bg-black px-4 py-2 text-sm text-white">创建并生成激活链接</button>
          </form>
        </Panel>
        <Panel title="发布统一价格">
          <form onSubmit={publish} className="grid gap-4">
            <input
              required
              placeholder="模型名称"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              className="rounded-lg border-gray-200"
            />
            <input
              type="number"
              min={0}
              value={inputRate}
              onChange={(event) => setInputRate(Number(event.target.value))}
              className="rounded-lg border-gray-200"
            />
            <input
              type="number"
              min={0}
              value={outputRate}
              onChange={(event) => setOutputRate(Number(event.target.value))}
              className="rounded-lg border-gray-200"
            />
            <button className="rounded-lg bg-black px-4 py-2 text-sm text-white">发布不可变版本</button>
          </form>
          <div className="mt-5 divide-y">
            {prices.map((price) => (
              <p key={price.price_id} className="py-3 text-sm">
                {price.model_name} · v{price.version}
              </p>
            ))}
          </div>
        </Panel>
      </div>
      <Panel title="企业账户" className="mt-6">
        {accounts.map((account) => (
          <div
            key={account.account_id}
            className="flex flex-wrap items-center justify-between gap-3 border-b py-4 last:border-0"
          >
            <div>
              <p className="font-medium">{account.company_name}</p>
              <p className="mt-1 text-xs text-gray-500">
                {account.email} · {account.available.display} · {account.status}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {account.status === "INVITED" && (
                <Action onClick={() => issueLink(account, "activation-link")}>激活链接</Action>
              )}
              {account.status === "ACTIVE" && (
                <Action onClick={() => issueLink(account, "password-reset-link")}>重置链接</Action>
              )}
              <Action onClick={() => adjust(account)}>调整额度</Action>
              {account.status !== "INVITED" && (
                <Action onClick={() => toggle(account)}>{account.status === "ACTIVE" ? "冻结" : "启用"}</Action>
              )}
            </div>
          </div>
        ))}
      </Panel>
    </div>
  );
}

function Panel({ title, children, className = "" }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-gray-200 bg-white p-6 ${className}`}>
      <h2 className="mb-5 text-lg font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Action({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button onClick={onClick} className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs">
      {children}
    </button>
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
