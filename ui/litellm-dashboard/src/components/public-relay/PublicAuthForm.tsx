"use client";

import Link from "next/link";
import { FormEvent, useCallback, useState } from "react";
import { useRouter } from "next/navigation";

import Turnstile from "./Turnstile";
import { relayFetch, SessionResult } from "@/lib/http/publicRelay";
import { migratedHref } from "@/utils/migratedPages";

type Mode = "login" | "register" | "reset";
const titles: Record<Mode, string> = {
  login: "Welcome back.",
  register: "Create your relay account.",
  reset: "Reset access.",
};
const submitLabels: Record<Mode, string> = {
  login: "Sign in",
  register: "Create account",
  reset: "Reset password",
};
const submitEndpoints: Record<Mode, string> = {
  login: "login",
  register: "register",
  reset: "password-reset",
};

export default function PublicAuthForm({ mode }: { mode: Mode }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [turnstileToken, setTurnstileToken] = useState("");
  const [resetKey, setResetKey] = useState(0);
  const [codeSent, setCodeSent] = useState(false);
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const handleToken = useCallback((token: string) => setTurnstileToken(token), []);
  const title = titles[mode];

  async function sendCode() {
    if (!turnstileToken) {
      setError("Complete the human verification first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const endpoint = mode === "register" ? "register/code" : "password-reset/code";
      await relayFetch(`/v1/public/auth/${endpoint}`, {
        method: "POST",
        body: JSON.stringify({ email, turnstile_token: turnstileToken }),
      });
      setCodeSent(true);
      setResetKey((value) => value + 1);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to send a code");
    } finally {
      setBusy(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!turnstileToken) {
      setError("Complete the human verification first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const endpoint = submitEndpoints[mode];
      const payload =
        mode === "login"
          ? { email, password, turnstile_token: turnstileToken }
          : { email, password, code, turnstile_token: turnstileToken };
      const { data } = await relayFetch<SessionResult | { message: string }>(`/v1/public/auth/${endpoint}`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      if (mode === "register" && "default_key" in data && data.default_key?.key) {
        setCreatedKey(data.default_key.key);
        return;
      }
      router.push(migratedHref(mode === "reset" ? "relay-login" : "portal"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to authenticate");
      setResetKey((value) => value + 1);
    } finally {
      setBusy(false);
    }
  }

  if (createdKey) {
    return (
      <main className="mx-auto grid min-h-[72vh] max-w-2xl place-items-center px-5 py-16">
        <section className="w-full rounded-3xl border border-black/10 bg-white p-8 shadow-sm">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-[#d83a20]">Account ready</p>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight">Copy your first key now.</h1>
          <p className="mt-3 text-sm leading-6 text-black/60">For security, the full value is shown only once.</p>
          <code className="mt-6 block overflow-x-auto rounded-2xl bg-[#171713] p-5 text-sm text-white">
            {createdKey}
          </code>
          <button
            type="button"
            onClick={() => navigator.clipboard.writeText(createdKey)}
            className="mt-4 rounded-full border border-black/20 px-5 py-2 text-sm"
          >
            Copy key
          </button>
          <Link
            href={migratedHref("portal")}
            className="ml-3 inline-block rounded-full bg-[#ff4f2e] px-5 py-2 text-sm text-white"
          >
            Open portal
          </Link>
        </section>
      </main>
    );
  }

  return (
    <main className="mx-auto grid min-h-[72vh] max-w-6xl gap-10 px-5 py-16 lg:grid-cols-2 lg:px-8">
      <div className="pt-8">
        <p className="font-mono text-xs uppercase tracking-[0.22em] text-[#d83a20]">Public relay</p>
        <h1 className="mt-5 max-w-lg text-5xl font-semibold tracking-[-0.045em]">{title}</h1>
        <p className="mt-5 max-w-md leading-7 text-black/60">
          Your portal session stays in a secure HTTP-only cookie. API keys are never used to sign into this page.
        </p>
      </div>
      <form onSubmit={submit} className="rounded-3xl border border-black/10 bg-white p-7 shadow-sm sm:p-9">
        <label className="block text-sm font-medium">
          Email
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="mt-2 w-full rounded-xl border-black/15"
          />
        </label>
        <label className="mt-5 block text-sm font-medium">
          Password
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
        {mode !== "login" && (
          <div className="mt-5">
            <div className="flex items-end gap-3">
              <label className="flex-1 text-sm font-medium">
                Six-digit code
                <input
                  inputMode="numeric"
                  pattern="[0-9]{6}"
                  required
                  value={code}
                  onChange={(event) => setCode(event.target.value)}
                  className="mt-2 w-full rounded-xl border-black/15"
                />
              </label>
              <button
                type="button"
                disabled={busy || !email || !turnstileToken}
                onClick={sendCode}
                className="mb-0.5 rounded-xl border border-black/15 px-4 py-2.5 text-sm disabled:opacity-40"
              >
                {codeSent ? "Send again" : "Send code"}
              </button>
            </div>
            {codeSent && (
              <p className="mt-2 text-xs text-black/50">Check your inbox, then complete verification again.</p>
            )}
          </div>
        )}
        <div className="mt-6">
          <Turnstile onToken={handleToken} resetKey={resetKey} />
        </div>
        {error && (
          <p role="alert" className="mt-4 text-sm text-red-700">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy || !turnstileToken}
          className="mt-6 w-full rounded-full bg-[#11110f] px-5 py-3 font-medium text-white disabled:opacity-40"
        >
          {busy ? "Working…" : submitLabels[mode]}
        </button>
        <div className="mt-5 flex justify-between text-sm text-black/55">
          {mode === "login" ? (
            <Link href={migratedHref("forgot-password")}>Forgot password?</Link>
          ) : (
            <Link href={migratedHref("relay-login")}>Back to sign in</Link>
          )}
          {mode === "login" && <Link href={migratedHref("register")}>Create account</Link>}
        </div>
      </form>
    </main>
  );
}
