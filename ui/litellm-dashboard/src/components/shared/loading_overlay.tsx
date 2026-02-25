import React, { useEffect, useRef, useState } from "react";
import { cx } from "@/lib/cva.config";
import { UiLoadingSpinner } from "../ui/ui-loading-spinner";

export type LoadingOverlayVariant = "overlay" | "subtle";

interface LoadingOverlayProps {
  loading: boolean;
  children: React.ReactNode;
  message?: string;
  /**
   * - "overlay": Full overlay with blurred content, blocks interaction. Use when content should not be used during load.
   * - "subtle": Keeps old data clearly visible with a small corner indicator. Use to show "data refreshing" without obscuring content.
   */
  variant?: LoadingOverlayVariant;
  /** Minimum time (ms) to show the loader once it appears. Prevents flicker on fast API responses. */
  minDisplayMs?: number;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  loading,
  children,
  message,
  variant = "overlay",
  minDisplayMs = 1000,
}) => {
  const isOverlay = variant === "overlay";
  const [visibleLoading, setVisibleLoading] = useState(loading);
  const loadingStartRef = useRef<number | null>(null);

  useEffect(() => {
    if (loading) {
      loadingStartRef.current = Date.now();
      setVisibleLoading(true);
    } else if (visibleLoading) {
      const elapsed = Date.now() - (loadingStartRef.current ?? 0);
      const remaining = Math.max(0, minDisplayMs - elapsed);
      if (remaining <= 0) {
        setVisibleLoading(false);
        loadingStartRef.current = null;
      } else {
        const timer = setTimeout(() => {
          setVisibleLoading(false);
          loadingStartRef.current = null;
        }, remaining);
        return () => clearTimeout(timer);
      }
    }
  }, [loading, minDisplayMs, visibleLoading]);

  return (
    <div className="relative">
      <div
        className={cx(
          "transition-all duration-200",
          visibleLoading && isOverlay && "blur-[2px] opacity-90 select-none pointer-events-none"
        )}
      >
        {children}
      </div>
      {visibleLoading && (
        <div
          data-testid="loading-overlay"
          className={cx(
            "absolute flex items-center gap-4",
            isOverlay ? "pointer-events-auto" : "pointer-events-none",
            isOverlay
              ? "inset-0 justify-center bg-white/30 dark:bg-gray-900/30"
              : "top-3 right-3 px-3 py-2 rounded-lg bg-white/90 dark:bg-gray-800/90 shadow-sm border border-gray-200/50 dark:border-gray-700/50"
          )}
          aria-busy="true"
          aria-live="polite"
        >
          <UiLoadingSpinner
            className={cx(
              "text-gray-600 dark:text-gray-400 shrink-0",
              isOverlay ? "size-16" : "size-6"
            )}
          />
          {message && (
            <span
              className={cx(
                "text-gray-600 dark:text-gray-400 whitespace-nowrap",
                isOverlay ? "text-2xl font-medium" : "text-sm"
              )}
            >
              {message}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default LoadingOverlay;
