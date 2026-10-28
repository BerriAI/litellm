import { describe, expect, it } from "vitest";

import { deriveKeyModelScope } from "./key_scope";

describe("deriveKeyModelScope", () => {
  it("treats unrestricted keys (null/empty allowed_routes) as full model access", () => {
    expect(deriveKeyModelScope(null)).toEqual({ hasModelAccess: true, label: null });
    expect(deriveKeyModelScope(undefined)).toEqual({ hasModelAccess: true, label: null });
    expect(deriveKeyModelScope([])).toEqual({ hasModelAccess: true, label: null });
  });

  it("classifies SCIM keys as no model access", () => {
    expect(deriveKeyModelScope(["/scim/*"])).toEqual({ hasModelAccess: false, label: "SCIM" });
    expect(deriveKeyModelScope(["/scim/v2/Users", "/scim/v2/Groups"])).toEqual({
      hasModelAccess: false,
      label: "SCIM",
    });
  });

  it("classifies management-only keys as no model access", () => {
    expect(deriveKeyModelScope(["management_routes"])).toEqual({ hasModelAccess: false, label: "Management" });
  });

  it("classifies read-only keys as no model access", () => {
    expect(deriveKeyModelScope(["info_routes"])).toEqual({ hasModelAccess: false, label: "Read-only" });
  });

  it("leaves LLM-API and custom scopes with model access (default rendering)", () => {
    expect(deriveKeyModelScope(["llm_api_routes"])).toEqual({ hasModelAccess: true, label: null });
    expect(deriveKeyModelScope(["/chat/completions"])).toEqual({ hasModelAccess: true, label: null });
    expect(deriveKeyModelScope(["management_routes", "llm_api_routes"])).toEqual({
      hasModelAccess: true,
      label: null,
    });
  });

  it("prefers a persisted key_type over allowed_routes for the no-inference buckets", () => {
    expect(deriveKeyModelScope([], "management")).toEqual({ hasModelAccess: false, label: "Management" });
    expect(deriveKeyModelScope([], "read_only")).toEqual({ hasModelAccess: false, label: "Read-only" });
    expect(deriveKeyModelScope(["some_future_mgmt_preset"], "management")).toEqual({
      hasModelAccess: false,
      label: "Management",
    });
  });

  it("falls back to allowed_routes for null/default/llm_api key_type", () => {
    expect(deriveKeyModelScope(["/scim/*"], null)).toEqual({ hasModelAccess: false, label: "SCIM" });
    expect(deriveKeyModelScope(["/scim/*"], "default")).toEqual({ hasModelAccess: false, label: "SCIM" });
    expect(deriveKeyModelScope([], "default")).toEqual({ hasModelAccess: true, label: null });
    expect(deriveKeyModelScope([], "llm_api")).toEqual({ hasModelAccess: true, label: null });
  });
});
