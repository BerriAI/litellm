"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState } from "react";

import { Account, relayFetch } from "@/lib/http/publicRelay";
import { migratedHref } from "@/utils/migratedPages";

type PortalValue = {
  account: Account;
  csrfToken: string;
};

const PortalContext = createContext<PortalValue | null>(null);

const navigation = [
  ["portal", "Overview"],
  ["portal/keys", "API keys"],
  ["portal/credit", "额度与账目"],
  ["portal/usage", "Usage"],
  ["portal/logs", "Request logs"],
  ["portal/security", "Security"],
];

export function PortalShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [value, setValue] = useState<PortalValue | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    relayFetch<Account>("/v1/portal/me")
      .then(({ data, response }) => {
        const csrfToken = response.headers.get("x-csrf-token");
        if (!csrfToken) {
          throw new Error("Portal session is missing CSRF protection");
        }
        setValue({ account: data, csrfToken });
      })
      .catch((requestError: Error) => {
        setError(requestError.message);
        router.replace(migratedHref("relay-login"));
      });
  }, [router]);

  if (!value) {
    return (
      <main className="grid min-h-screen place-items-center bg-[#f4f4ef]">
        <p className="text-sm text-black/50">{error || "Loading your portal…"}</p>
      </main>
    );
  }

  return (
    <PortalContext.Provider value={value}>
      <div className="min-h-screen bg-[#f4f4ef] text-[#11110f] lg:grid lg:grid-cols-[250px_1fr]">
        <aside className="border-b border-black/10 bg-[#171713] p-5 text-white lg:min-h-screen lg:border-b-0">
          <Link href={migratedHref("")} className="flex items-center gap-3 font-semibold">
            <span className="grid size-8 place-items-center rounded-full bg-[#ff4f2e] text-sm">L</span>
            LiteLLM Relay
          </Link>
          <nav className="mt-10 grid grid-cols-2 gap-1 text-sm sm:grid-cols-3 lg:grid-cols-1">
            {navigation.map(([route, label]) => {
              const href = migratedHref(route);
              return (
                <Link
                  key={href}
                  href={href}
                  className={`rounded-xl px-3 py-2.5 ${pathname === href ? "bg-white text-black" : "text-white/65 hover:bg-white/10"}`}
                >
                  {label}
                </Link>
              );
            })}
          </nav>
          <p className="mt-10 hidden break-all text-xs text-white/35 lg:block">{value.account.email}</p>
        </aside>
        <main className="min-w-0 p-5 sm:p-8 lg:p-10">{children}</main>
      </div>
    </PortalContext.Provider>
  );
}

export function usePortal(): PortalValue {
  const value = useContext(PortalContext);
  if (!value) {
    throw new Error("Portal context is unavailable");
  }
  return value;
}
