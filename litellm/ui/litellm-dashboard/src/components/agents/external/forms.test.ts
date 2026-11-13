import { expect, it } from "vitest";
import { pageParser } from "./forms";

it.each(["0", "-1", "1.5", "junk", "9999999999999999999999"])("rejects invalid page query %s", (value) => {
  expect(pageParser.parse(value)).toBeNull();
});
