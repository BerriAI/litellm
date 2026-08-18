import { isValidElement, type ReactNode } from "react";
import { toast, type ToastKind } from "@/lib/toast";

export type NotificationConfig = {
  readonly message?: ReactNode;
  readonly description?: ReactNode;
  readonly duration?: number;
  readonly placement?: string;
  readonly key?: string;
};

type NotificationInput = ReactNode | NotificationConfig;

const FALLBACK_TITLES: Readonly<Record<ToastKind, string>> = {
  success: "Success",
  info: "Info",
  warning: "Warning",
  error: "Error",
};

const secondsToMs = (seconds: number | undefined): number | undefined =>
  seconds === undefined ? undefined : seconds * 1000;

const isConfig = (input: NotificationInput): input is NotificationConfig =>
  input !== null && typeof input === "object" && !isValidElement(input) && !(Symbol.iterator in input);

const show = (kind: ToastKind, input: NotificationInput): void => {
  if (!isConfig(input)) {
    toast[kind](input);
    return;
  }
  toast[kind](input.message ?? FALLBACK_TITLES[kind], {
    description: input.description,
    durationMs: secondsToMs(input.duration),
  });
};

const NotificationManager = {
  success: (input: NotificationInput): void => show("success", input),
  info: (input: NotificationInput): void => show("info", input),
  warning: (input: NotificationInput): void => show("warning", input),
  error: (input: NotificationInput): void => show("error", input),
  fromBackend: (input: unknown, extra?: Omit<NotificationConfig, "message" | "description">): void =>
    toast.fromError(input, { durationMs: secondsToMs(extra?.duration) }),
  clear: (): void => toast.dismiss(),
};

export default NotificationManager;
