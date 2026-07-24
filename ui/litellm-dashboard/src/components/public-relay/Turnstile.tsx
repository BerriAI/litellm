"use client";

import Script from "next/script";
import { useCallback, useEffect, useRef } from "react";

declare global {
  interface Window {
    turnstile?: {
      render: (element: HTMLElement, options: { sitekey: string; callback: (token: string) => void }) => string;
      remove: (widgetId: string) => void;
    };
  }
}

export default function Turnstile({ onToken, resetKey }: { onToken: (token: string) => void; resetKey: number }) {
  const elementRef = useRef<HTMLDivElement>(null);
  const widgetRef = useRef<string | null>(null);
  const siteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;

  const renderWidget = useCallback(() => {
    if (!siteKey || !window.turnstile || !elementRef.current || widgetRef.current) {
      return;
    }
    widgetRef.current = window.turnstile.render(elementRef.current, { sitekey: siteKey, callback: onToken });
  }, [onToken, siteKey]);

  useEffect(() => {
    if (widgetRef.current && window.turnstile) {
      window.turnstile.remove(widgetRef.current);
      widgetRef.current = null;
      onToken("");
    }
    renderWidget();
  }, [onToken, renderWidget, resetKey]);

  if (!siteKey) {
    return <p className="rounded-xl bg-amber-50 p-3 text-sm text-amber-800">Turnstile site key is not configured.</p>;
  }

  return (
    <>
      <Script src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit" onLoad={renderWidget} />
      <div ref={elementRef} className="min-h-16" />
    </>
  );
}
