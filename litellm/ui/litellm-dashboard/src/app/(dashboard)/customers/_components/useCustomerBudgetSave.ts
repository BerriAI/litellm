import { useRef, useState } from "react";
import { extractProxyErrorMessage } from "@/lib/http/client";

export function useCustomerBudgetSave(onPendingChange?: (pending: boolean) => void) {
  const saving = useRef(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (write: () => Promise<void>) => {
    if (saving.current) return;
    saving.current = true;
    setPending(true);
    onPendingChange?.(true);
    setError(null);
    try {
      await write();
    } catch (cause) {
      setError(extractProxyErrorMessage(cause));
    } finally {
      saving.current = false;
      setPending(false);
      onPendingChange?.(false);
    }
  };
  return { pending, error, setError, run };
}
