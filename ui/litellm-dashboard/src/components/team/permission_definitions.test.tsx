import { describe, expect, it } from "vitest";
import { getMethodForEndpoint, getPermissionInfo, PERMISSION_DESCRIPTIONS } from "./permission_definitions";

describe("permission_definitions", () => {
  describe("getMethodForEndpoint", () => {
    it("should return GET for info endpoints", () => {
      expect(getMethodForEndpoint("/key/info")).toBe("GET");
    });

    it("should return GET for list endpoints", () => {
      expect(getMethodForEndpoint("/key/list")).toBe("GET");
    });

    it("should return POST for other endpoints", () => {
      expect(getMethodForEndpoint("/key/generate")).toBe("POST");
      expect(getMethodForEndpoint("/key/update")).toBe("POST");
      expect(getMethodForEndpoint("/key/delete")).toBe("POST");
    });
  });

  describe("getPermissionInfo", () => {
    it("should return correct info for exact match permission", () => {
      const result = getPermissionInfo("/key/generate");
      expect(result.method).toBe("POST");
      expect(result.endpoint).toBe("/key/generate");
      expect(result.description).toBe(PERMISSION_DESCRIPTIONS["/key/generate"]);
      expect(result.route).toBe("/key/generate");
    });

    it("should return GET method for info endpoint", () => {
      const result = getPermissionInfo("/key/info");
      expect(result.method).toBe("GET");
      expect(result.endpoint).toBe("/key/info");
      expect(result.description).toBe(PERMISSION_DESCRIPTIONS["/key/info"]);
    });

    it("should return GET method for list endpoint", () => {
      const result = getPermissionInfo("/key/list");
      expect(result.method).toBe("GET");
      expect(result.endpoint).toBe("/key/list");
      expect(result.description).toBe(PERMISSION_DESCRIPTIONS["/key/list"]);
    });

    it("should find partial match for permission with pattern", () => {
      const result = getPermissionInfo("/key/service-account/generate");
      expect(result.method).toBe("POST");
      expect(result.endpoint).toBe("/key/service-account/generate");
      expect(result.description).toBe(PERMISSION_DESCRIPTIONS["/key/service-account/generate"]);
    });

    it("should return fallback description for unknown permission", () => {
      const result = getPermissionInfo("/unknown/endpoint");
      expect(result.method).toBe("POST");
      expect(result.endpoint).toBe("/unknown/endpoint");
      expect(result.description).toBe("Access /unknown/endpoint");
      expect(result.route).toBe("/unknown/endpoint");
    });
  });
});
