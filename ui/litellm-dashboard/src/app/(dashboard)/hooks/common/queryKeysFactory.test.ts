import { describe, it, expect } from "vitest";
import { createQueryKeys } from "./queryKeysFactory";

describe("createQueryKeys", () => {
  const keys = createQueryKeys("books");

  it("should return the resource name as the base key", () => {
    expect(keys.all).toEqual(["books"]);
  });

  it("should generate a lists key", () => {
    expect(keys.lists()).toEqual(["books", "list"]);
  });

  it("should generate a list key with params", () => {
    expect(keys.list({ page: 1, limit: 10 })).toEqual([
      "books",
      "list",
      { params: { page: 1, limit: 10 } },
    ]);
  });

  it("should generate a list key with undefined params when none provided", () => {
    expect(keys.list()).toEqual(["books", "list", { params: undefined }]);
  });

  it("should generate a details key", () => {
    expect(keys.details()).toEqual(["books", "detail"]);
  });

  it("should generate a detail key for a specific ID", () => {
    expect(keys.detail("123")).toEqual(["books", "detail", "123"]);
  });
});
