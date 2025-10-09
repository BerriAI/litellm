"use client";

const getBasePath = () => {
  const raw = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = raw.replace(/^\/+|\/+$/g, ""); // strip leading/trailing slashes
  return trimmed ? `/${trimmed}/` : "/"; // ensure trailing slash
};

import React, { createContext, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation"; // ⟵ add this

type Flags = {
  refactoredUIFlag: boolean;
  setRefactoredUIFlag: (v: boolean) => void;
};

const STORAGE_KEY = "feature.refactoredUIFlag";

const FeatureFlagsCtx = createContext<Flags | null>(null);

/** Safely read the flag from localStorage. If anything goes wrong, reset to false. */
function readFlagSafely(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) {
      localStorage.setItem(STORAGE_KEY, "false");
      return false;
    }

    const v = raw.trim().toLowerCase();
    if (v === "true" || v === "1") return true;
    if (v === "false" || v === "0") return false;

    // Last chance: try JSON.parse in case something odd was stored.
    const parsed = JSON.parse(raw);
    if (typeof parsed === "boolean") return parsed;

    // Malformed → reset to false
    localStorage.setItem(STORAGE_KEY, "false");
    return false;
  } catch {
    // If even accessing localStorage throws, best effort reset then default to false
    try {
      localStorage.setItem(STORAGE_KEY, "false");
    } catch {}
    return false;
  }
}

function writeFlagSafely(v: boolean) {
  try {
    localStorage.setItem(STORAGE_KEY, String(v));
  } catch {
    // Ignore write errors; state will still reflect the intended value.
  }
}

export const FeatureFlagsProvider = ({ children }: { children: React.ReactNode }) => {
  const router = useRouter(); // ⟵ add this

  // Lazy init reads from localStorage only on the client
  const [refactoredUIFlag, setRefactoredUIFlagState] = useState<boolean>(() => readFlagSafely());

  const setRefactoredUIFlag = (v: boolean) => {
    setRefactoredUIFlagState(v);
    writeFlagSafely(v);
  };

  // Keep this flag in sync across tabs/windows.
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue != null) {
        const next = e.newValue.trim().toLowerCase();
        setRefactoredUIFlagState(next === "true" || next === "1");
      }
      // If the key was cleared elsewhere, self-heal to false.
      if (e.key === STORAGE_KEY && e.newValue === null) {
        writeFlagSafely(false);
        setRefactoredUIFlagState(false);
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Redirect to base path the moment the flag is OFF.
  useEffect(() => {
    if (refactoredUIFlag) return; // only act when turned off

    const base = getBasePath();
    const normalize = (p: string) => (p.endsWith("/") ? p : p + "/");
    const current = normalize(window.location.pathname);

    // Avoid a redirect loop if we're already at the base path.
    if (current !== base) {
      // Replace so the "off" redirect doesn't pollute history.
      router.replace(base);
    }
  }, [refactoredUIFlag, router]);

  return (
    <FeatureFlagsCtx.Provider value={{ refactoredUIFlag, setRefactoredUIFlag }}>{children}</FeatureFlagsCtx.Provider>
  );
};

const useFeatureFlags = () => {
  const ctx = useContext(FeatureFlagsCtx);
  if (!ctx) throw new Error("useFeatureFlags must be used within FeatureFlagsProvider");
  return ctx;
};

export default useFeatureFlags;
