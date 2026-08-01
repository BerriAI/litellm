"use client";

import React, { useEffect, useSyncExternalStore } from "react";
import { ResetPasswordForm } from "./ResetPasswordForm";

function subscribeToHashChange(callback: () => void) {
  window.addEventListener("hashchange", callback);
  return () => window.removeEventListener("hashchange", callback);
}

function getTokenFromFragment(): string | null {
  const hash = window.location.hash.replace(/^#/, "");
  return new URLSearchParams(hash).get("token");
}

function getServerTokenSnapshot(): string | null {
  return null;
}

/**
 * Reads the reset token from the URL fragment (not a `?token=` query param, which
 * would be sent to the server and land in access logs / Referer headers).
 */
function useTokenFromFragment(): string | null {
  return useSyncExternalStore(subscribeToHashChange, getTokenFromFragment, getServerTokenSnapshot);
}

export default function ResetPassword() {
  const token = useTokenFromFragment();

  useEffect(() => {
    // Strip the token from the visible URL once read, so it doesn't linger in
    // browser history.
    if (token) {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  }, [token]);

  return <ResetPasswordForm token={token} />;
}
