import { useState, type FormEvent } from "react";
import { useCustomerBudgetSave } from "./useCustomerBudgetSave";

export type BudgetFormResult<T> = { ok: true; value: T } | { ok: false; error: string };

export function useCustomerBudgetForm<Fields, Payload>(
  {
    initial,
    parse,
    write,
  }: {
    initial: Fields;
    parse: (fields: Fields) => BudgetFormResult<Payload>;
    write: (payload: Payload) => Promise<void>;
  },
  onPendingChange?: (pending: boolean) => void,
) {
  const [fields, setFields] = useState(initial);
  const { pending, error, setError, run } = useCustomerBudgetSave(onPendingChange);
  const update = <K extends keyof Fields>(key: K, value: Fields[K]) => {
    setFields((current) => ({ ...current, [key]: value }));
    setError(null);
  };
  const reset = () => {
    setFields(initial);
    setError(null);
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (pending) return;
    const result = parse(fields);
    if (!result.ok) return setError(result.error);
    await run(() => write(result.value));
  };
  return {
    fields,
    setFields,
    update,
    reset,
    pending,
    error,
    setError,
    save,
    dirty: JSON.stringify(fields) !== JSON.stringify(initial),
  };
}
