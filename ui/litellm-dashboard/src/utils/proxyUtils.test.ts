import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchProxySettings } from "./proxyUtils";
import { getProxyUISettings } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getProxyUISettings: vi.fn(),
}));

describe("fetchProxySettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should return null when accessToken is null", async () => {
    const result = await fetchProxySettings(null);

    expect(result).toBeNull();
    expect(getProxyUISettings).not.toHaveBeenCalled();
  });

  it("should return null when accessToken is undefined", async () => {
    const result = await fetchProxySettings(undefined as any);

    expect(result).toBeNull();
    expect(getProxyUISettings).not.toHaveBeenCalled();
  });

  it("should return proxy settings when getProxyUISettings succeeds", async () => {
    const mockProxySettings = { someSetting: "value", anotherSetting: 123 };
    const accessToken = "test-token";

    vi.mocked(getProxyUISettings).mockResolvedValue(mockProxySettings);

    const result = await fetchProxySettings(accessToken);

    expect(result).toEqual(mockProxySettings);
    expect(getProxyUISettings).toHaveBeenCalledOnce();
    expect(getProxyUISettings).toHaveBeenCalledWith(accessToken);
  });

  it("should return null and log error when getProxyUISettings throws", async () => {
    const accessToken = "test-token";
    const mockError = new Error("Network error");
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    vi.mocked(getProxyUISettings).mockRejectedValue(mockError);

    const result = await fetchProxySettings(accessToken);

    expect(result).toBeNull();
    expect(getProxyUISettings).toHaveBeenCalledOnce();
    expect(getProxyUISettings).toHaveBeenCalledWith(accessToken);
    expect(consoleSpy).toHaveBeenCalledWith("Error fetching proxy settings:", mockError);

    consoleSpy.mockRestore();
  });

  it("should return null and log error when getProxyUISettings throws a string", async () => {
    const accessToken = "test-token";
    const mockError = "String error";
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    vi.mocked(getProxyUISettings).mockRejectedValue(mockError);

    const result = await fetchProxySettings(accessToken);

    expect(result).toBeNull();
    expect(getProxyUISettings).toHaveBeenCalledOnce();
    expect(getProxyUISettings).toHaveBeenCalledWith(accessToken);
    expect(consoleSpy).toHaveBeenCalledWith("Error fetching proxy settings:", mockError);

    consoleSpy.mockRestore();
  });
});