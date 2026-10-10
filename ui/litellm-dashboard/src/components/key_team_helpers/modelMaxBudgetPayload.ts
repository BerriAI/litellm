import type { ModelMaxBudget } from "./ModelMaxBudgetEditor";

/**
 * A stored budget can carry either spelling: the proxy's BudgetConfig aliases
 * `budget_limit`/`time_period` onto `max_budget`/`budget_duration`, and its CRUD
 * endpoints accept the limit as a string.
 */
export type StoredModelMaxBudget = Record<
  string,
  | {
      budget_limit?: number | string | null;
      time_period?: string | null;
      max_budget?: number | string | null;
      budget_duration?: string | null;
    }
  | null
  | undefined
>;

const canonical = (budget: StoredModelMaxBudget | null | undefined): string =>
  JSON.stringify(
    Object.entries(budget ?? {})
      .map(([model, config]) => [
        model,
        Number(config?.budget_limit ?? config?.max_budget ?? NaN),
        config?.time_period ?? config?.budget_duration ?? null,
      ])
      .sort((left, right) => String(left[0]).localeCompare(String(right[0]))),
  );

/**
 * What to send for `model_max_budget` on an update, or undefined to omit the key.
 *
 * Omitting an unchanged budget matters beyond saving bytes: the write is gated on
 * an enterprise license, so re-sending what is already stored makes an unrelated
 * edit fail with a 400 on a proxy without one.
 *
 * Clearing the last row still has to send `{}`: omitting the key leaves the stored
 * budgets in place, so the row the operator just deleted would keep enforcing.
 */
export const modelMaxBudgetUpdate = (
  edited: ModelMaxBudget,
  stored: StoredModelMaxBudget | null | undefined,
): ModelMaxBudget | undefined => (canonical(edited) === canonical(stored) ? undefined : edited);
