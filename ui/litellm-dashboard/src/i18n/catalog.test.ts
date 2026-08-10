import { describe, expect, it } from "vitest";
import { resources, TRANSLATION_NAMESPACES } from "./catalog";

const leafPaths = (value: object, prefix = ""): string[] =>
  Object.entries(value)
    .flatMap(([key, child]) => {
      const path = prefix ? `${prefix}.${key}` : key;

      return typeof child === "object" && child !== null ? leafPaths(child, path) : [path];
    })
    .sort();

describe("translation catalog", () => {
  it("registers the localized product namespaces", () => {
    expect(TRANSLATION_NAMESPACES).toEqual(["common", "auth", "navigation", "chat"]);
  });

  it.each(TRANSLATION_NAMESPACES)("keeps Russian %s keys in parity with English", (namespace) => {
    expect(leafPaths(resources.ru[namespace])).toEqual(leafPaths(resources.en[namespace]));
  });
});
