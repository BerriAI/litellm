import type { DestinationVariablesFields } from "./schema";

type DestinationVariables = Record<string, unknown>;

export const toDestinationFieldValue = (value: unknown) => (value == null ? "" : String(value));

// Values rendered from the server are masked or sourced from the process environment, so
// re-submitting an untouched field would persist a mask or copy env-managed config into the database.
export const toChangedVariables = (
  variables: DestinationVariables,
  fields: DestinationVariablesFields,
): Record<string, string> =>
  Object.fromEntries(
    Object.entries(variables).flatMap(([key, value]) => {
      const edited = fields[key];
      if (!edited || edited === toDestinationFieldValue(value)) {
        return [];
      }
      return [[key, edited] as const];
    }),
  );
