import { useState } from "react";

/**
 * State seeded from `seed`, re-seeded whenever `identity` changes.
 *
 * The per-model budget editor holds its rows in state seeded once on mount, so
 * it cannot re-read its `value` prop: that prop is the state it feeds, and
 * re-hydrating on every change would wipe a half-typed row. Loading a different
 * key or user therefore has to re-seed here, or the rows on screen keep
 * describing the previously loaded one and a save overwrites their budgets.
 *
 * Adjusting state during render is React's documented way to reset state on a
 * prop change. An effect would paint one frame with the stale value first, and
 * the dashboard's lint rules reject synchronous setState inside an effect.
 */
export function useSeededState<T>(identity: unknown, seed: () => T): [T, (next: T) => void] {
  const [value, setValue] = useState<T>(seed);
  const [seededFrom, setSeededFrom] = useState(identity);

  if (seededFrom !== identity) {
    setSeededFrom(identity);
    setValue(seed());
  }

  return [value, setValue];
}
