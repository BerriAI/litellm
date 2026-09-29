// Shared test helper for temporarily mutating process.env.

/** Temporarily set (undefined = unset) env vars; returns restore() to revert. */
export function setEnv(vars: Record<string, string | undefined>): { restore: () => void } {
  const prev: Record<string, string | undefined> = {};
  for (const [k, v] of Object.entries(vars)) {
    prev[k] = process.env[k];
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  return {
    restore() {
      for (const [k, v] of Object.entries(prev)) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
    },
  };
}
