/**
 * Global message manager — thin wrapper around sonner's `toast.*` API.
 *
 * Phase-1 shadcn migration: this file previously delegated to antd's
 * `message.*` API. It now delegates to sonner. The public API (including
 * `setMessageInstance`) is preserved so existing call sites don't need to
 * change; the `setMessageInstance` function is now a no-op kept for
 * backwards-compatibility (sonner is singleton-rendered at the root layout
 * and does not need per-tree instance registration).
 */
import { toast } from "sonner";

/**
 * No-op retained for API compatibility with call sites that import
 * `setMessageInstance` from this module. Sonner is rendered globally via
 * `<Toaster />` in the root layout; no per-tree registration is required.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export const setMessageInstance = (_instance: unknown) => {
  // no-op — sonner is singleton
};

/** antd durations are seconds; sonner durations are milliseconds. */
function toSonnerDuration(duration?: number): number | undefined {
  if (duration == null) return undefined;
  return duration * 1000;
}

const MessageManager = {
  success(content: string, duration?: number) {
    return toast.success(content, { duration: toSonnerDuration(duration) });
  },

  error(content: string, duration?: number) {
    return toast.error(content, { duration: toSonnerDuration(duration) });
  },

  warning(content: string, duration?: number) {
    return toast.warning(content, { duration: toSonnerDuration(duration) });
  },

  info(content: string, duration?: number) {
    return toast.info(content, { duration: toSonnerDuration(duration) });
  },

  /**
   * Show a loading toast. Returns the sonner toast id, which can be passed
   * to `toast.dismiss(id)` (or `MessageManager.destroy()` to clear all).
   * Behaves like antd's `message.loading` — duration in seconds; omit to
   * keep the toast open until dismissed.
   */
  loading(content: string, duration?: number) {
    return toast.loading(content, { duration: toSonnerDuration(duration) });
  },

  destroy() {
    toast.dismiss();
  },
};

export default MessageManager;
