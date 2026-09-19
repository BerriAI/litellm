import type { ModelMaxBudget } from "./ModelMaxBudgetEditor";
import { modelMaxBudgetUpdate, type StoredModelMaxBudget } from "./modelMaxBudgetPayload";
import { useSeededState } from "./useSeededState";

interface ModelMaxBudgetField<TValues> {
  readonly value: ModelMaxBudget;
  readonly setValue: (next: ModelMaxBudget) => void;
  /** Adds `model_max_budget` to the payload only when it actually changed. */
  readonly applyTo: (values: TValues) => void;
}

/**
 * The seeding and submit halves of the per-model budget field, kept together
 * because they share one invariant: both compare against the SAME stored value.
 * Seeding from one record while diffing against another is how an edit form
 * ends up writing the previously loaded key's budgets onto the current one.
 */
export function useModelMaxBudgetField<TValues extends { model_max_budget?: ModelMaxBudget }>(
  identity: unknown,
  stored: StoredModelMaxBudget | null | undefined,
): ModelMaxBudgetField<TValues> {
  const [value, setValue] = useSeededState<ModelMaxBudget>(identity, () => (stored ?? {}) as ModelMaxBudget);

  return {
    value,
    setValue,
    applyTo: (values: TValues) => {
      const update = modelMaxBudgetUpdate(value, stored);
      if (update !== undefined) {
        values.model_max_budget = update;
      }
    },
  };
}
