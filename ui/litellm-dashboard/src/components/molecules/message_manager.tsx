import { toast } from "@/lib/toast";

const secondsToMs = (seconds: number | undefined): number | undefined =>
  seconds === undefined ? undefined : seconds * 1000;

const MessageManager = {
  success: (content: string, duration?: number): void => toast.success(content, { durationMs: secondsToMs(duration) }),
  error: (content: string, duration?: number): void => toast.error(content, { durationMs: secondsToMs(duration) }),
  warning: (content: string, duration?: number): void => toast.warning(content, { durationMs: secondsToMs(duration) }),
  info: (content: string, duration?: number): void => toast.info(content, { durationMs: secondsToMs(duration) }),
  destroy: (): void => toast.dismiss(),
};

export default MessageManager;
