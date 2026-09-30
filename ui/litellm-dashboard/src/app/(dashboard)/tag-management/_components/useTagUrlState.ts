import { parseAsBoolean, parseAsString, useQueryStates } from "nuqs";

const TAG_URL_PARSERS = {
  tag: parseAsString.withOptions({ history: "push" }),
  edit: parseAsBoolean.withDefault(false),
};

export function useTagUrlState() {
  return useQueryStates(TAG_URL_PARSERS);
}
