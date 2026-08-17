import { toast } from "@/lib/toast";

const secondsToMs = (seconds: number | undefined): number | undefined =>
  seconds === undefined ? undefined : seconds * 1000;

/** Legacy alias for `toast` from `@/lib/toast`; durations are seconds for antd-era callers. */
const MessageManager = {
  success: (content: string, duration?: number): void => toast.success(content, { durationMs: secondsToMs(duration) }),
  error: (content: string, duration?: number): void => toast.error(content, { durationMs: secondsToMs(duration) }),
  warning: (content: string, duration?: number): void => toast.warning(content, { durationMs: secondsToMs(duration) }),
  info: (content: string, duration?: number): void => toast.info(content, { durationMs: secondsToMs(duration) }),
  destroy: (): void => toast.dismiss(),
};

export default MessageManager;
