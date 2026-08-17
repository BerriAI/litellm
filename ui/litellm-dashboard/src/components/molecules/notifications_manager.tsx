import type { ReactNode } from "react";
import { toast } from "@/lib/toast";

/** Legacy alias for `toast` from `@/lib/toast`; `fromBackend` is `toast.fromError`. */
const NotificationManager = {
  success: (message: ReactNode): void => toast.success(message),
  info: (message: ReactNode): void => toast.info(message),
  warning: (message: ReactNode): void => toast.warning(message),
  error: (message: ReactNode): void => toast.error(message),
  fromBackend: (input: unknown): void => toast.fromError(input),
  clear: (): void => toast.dismiss(),
};

export default NotificationManager;
