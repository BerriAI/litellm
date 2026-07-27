"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

import { relayFetch, SessionResult } from "@/lib/http/publicRelay";
import { migratedHref } from "@/utils/migratedPages";

type Mode = "login" | "activate" | "reset";

const titles: Record<Mode, string> = {
  login: "企业账户登录",
  activate: "激活企业账户",
  reset: "设置新密码",
};

export default function PublicAuthForm({ mode }: { mode: Mode }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [createdKey, setCreatedKey] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const buttonLabel = mode === "login" ? "登录" : "确认";

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const endpoint = mode === "reset" ? "password-reset" : mode;
      const token = new URLSearchParams(window.location.search).get("token") ?? "";
      const payload = mode === "login" ? { email, password } : { token, password };
      const { data } = await relayFetch<SessionResult | { message: string }>(`/v1/public/auth/${endpoint}`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (mode === "activate" && "default_key" in data && data.default_key?.key) {
        setCreatedKey(data.default_key.key);
        return;
      }
      router.push(migratedHref(mode === "reset" ? "relay-login" : "portal"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "操作失败");
    } finally {
      setBusy(false);
    }
  }

  if (createdKey) {
    return (
      <main className="mx-auto grid min-h-[72vh] max-w-2xl place-items-center px-5 py-16">
        <section className="w-full rounded-3xl border border-black/10 bg-white p-8 shadow-sm">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-[#d83a20]">账户已激活</p>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight">请立即保存默认 API Key</h1>
          <p className="mt-3 text-sm leading-6 text-black/60">完整 Key 只显示这一次，默认关闭请求内容记录。</p>
          <code className="mt-6 block overflow-x-auto rounded-2xl bg-[#171713] p-5 text-sm text-white">
            {createdKey}
          </code>
          <button
            type="button"
            onClick={() => navigator.clipboard.writeText(createdKey)}
            className="mt-4 rounded-full border border-black/20 px-5 py-2 text-sm"
          >
            复制 Key
          </button>
          <Link href={migratedHref("portal")} className="ml-3 rounded-full bg-[#ff4f2e] px-5 py-2 text-sm text-white">
            进入企业门户
          </Link>
        </section>
      </main>
    );
  }

  return (
    <main className="mx-auto grid min-h-[72vh] max-w-5xl gap-10 px-5 py-16 lg:grid-cols-2 lg:px-8">
      <div className="pt-8">
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-[#d83a20]">Enterprise relay</p>
        <h1 className="mt-5 text-5xl font-semibold tracking-[-0.045em]">{titles[mode]}</h1>
        <p className="mt-5 max-w-md leading-7 text-black/60">
          账户由管理员开通。激活和密码重置链接均为一次性链接，门户会话保存在安全 Cookie 中。
        </p>
      </div>
      <form onSubmit={submit} className="rounded-3xl border border-black/10 bg-white p-7 shadow-sm sm:p-9">
        {mode === "login" && (
          <label className="block text-sm font-medium">
            管理员邮箱
            <input
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="mt-2 w-full rounded-xl border-black/15"
            />
          </label>
        )}
        <label className={`${mode === "login" ? "mt-5" : ""} block text-sm font-medium`}>
          {mode === "login" ? "密码" : "新密码"}
          <input
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={12}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="mt-2 w-full rounded-xl border-black/15"
          />
        </label>
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="mt-6 w-full rounded-full bg-[#11110f] px-5 py-3 font-medium text-white disabled:opacity-40"
        >
          {busy ? "处理中…" : buttonLabel}
        </button>
        {mode !== "login" && (
          <Link href={migratedHref("relay-login")} className="mt-5 block text-sm text-black/55">
            返回登录
          </Link>
        )}
      </form>
    </main>
  );
}
