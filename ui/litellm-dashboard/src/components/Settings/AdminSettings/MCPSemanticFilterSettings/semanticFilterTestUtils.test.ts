import { describe, it, expect, vi, beforeEach } from "vitest";
import { getCurlCommand, runSemanticFilterTest } from "./semanticFilterTestUtils";
import { testMCPSemanticFilter } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";

vi.mock("@/components/networking", () => ({
  testMCPSemanticFilter: vi.fn(),
}));

describe("getCurlCommand", () => {
  it("should include the model name in the curl command", () => {
    const result = getCurlCommand("gpt-4o", "test query");
    expect(result).toContain('"gpt-4o"');
  });

  it("should include the query in the curl command", () => {
    const result = getCurlCommand("gpt-4o", "find relevant files");
    expect(result).toContain("find relevant files");
  });

  it("should use a placeholder when query is empty", () => {
    const result = getCurlCommand("gpt-4o", "");
    expect(result).toContain("Your query here");
  });
});

describe("runSemanticFilterTest", () => {
  const mockSetIsTesting = vi.fn();
  const mockSetTestResult = vi.fn();
  const baseArgs = {
    accessToken: "test-token",
    testModel: "gpt-4o",
    testQuery: "find relevant files",
    setIsTesting: mockSetIsTesting,
    setTestResult: mockSetTestResult,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call NotificationManager.error and not set isTesting when testQuery is empty", async () => {
    await runSemanticFilterTest({ ...baseArgs, testQuery: "" });
    expect(NotificationManager.error).toHaveBeenCalledWith("Please enter a query and select a model");
    expect(mockSetIsTesting).not.toHaveBeenCalled();
  });

  it("should call NotificationManager.error and not set isTesting when testModel is empty", async () => {
    await runSemanticFilterTest({ ...baseArgs, testModel: "" });
    expect(NotificationManager.error).toHaveBeenCalledWith("Please enter a query and select a model");
    expect(mockSetIsTesting).not.toHaveBeenCalled();
  });

  it("should set isTesting to true then false around the API call", async () => {
    vi.mocked(testMCPSemanticFilter).mockResolvedValueOnce({
      data: {},
      headers: { filter: "5->2", tools: "tool-a,tool-b" },
    });

    await runSemanticFilterTest(baseArgs);

    expect(mockSetIsTesting).toHaveBeenNthCalledWith(1, true);
    expect(mockSetIsTesting).toHaveBeenNthCalledWith(2, false);
  });

  it("should clear the previous test result before making a new request", async () => {
    vi.mocked(testMCPSemanticFilter).mockResolvedValueOnce({
      data: {},
      headers: { filter: "5->2", tools: "tool-a,tool-b" },
    });

    await runSemanticFilterTest(baseArgs);

    expect(mockSetTestResult).toHaveBeenNthCalledWith(1, null);
  });

  it("should set test result with parsed data on success", async () => {
    vi.mocked(testMCPSemanticFilter).mockResolvedValueOnce({
      data: {},
      headers: { filter: "10->3", tools: "wiki,github,slack" },
    });

    await runSemanticFilterTest(baseArgs);

    expect(mockSetTestResult).toHaveBeenCalledWith({
      totalTools: 10,
      selectedTools: 3,
      tools: ["wiki", "github", "slack"],
    });
    expect(NotificationManager.success).toHaveBeenCalledWith(
      "Semantic filter test completed successfully"
    );
  });

  it("should show a warning when the filter header is missing", async () => {
    vi.mocked(testMCPSemanticFilter).mockResolvedValueOnce({
      data: {},
      headers: { filter: null, tools: null },
    });

    await runSemanticFilterTest(baseArgs);

    expect(NotificationManager.warning).toHaveBeenCalledWith(
      "Semantic filter is not enabled or no tools were filtered"
    );
    expect(mockSetTestResult).not.toHaveBeenCalledWith(expect.objectContaining({ totalTools: expect.any(Number) }));
  });

  it("should show an error notification and finish testing when the API call fails", async () => {
    vi.mocked(testMCPSemanticFilter).mockRejectedValueOnce(new Error("Network error"));

    await runSemanticFilterTest(baseArgs);

    expect(NotificationManager.error).toHaveBeenCalledWith("Failed to test semantic filter");
    expect(mockSetIsTesting).toHaveBeenLastCalledWith(false);
  });
});
