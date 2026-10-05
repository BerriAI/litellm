import { createParser } from "nuqs";

export const parseAsDemoFlag = createParser({
  parse: (value) => (value === "1" ? true : null),
  serialize: () => "1",
});
