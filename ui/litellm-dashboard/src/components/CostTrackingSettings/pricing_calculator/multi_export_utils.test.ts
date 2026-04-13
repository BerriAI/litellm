import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { exportMultiToPDF, exportMultiToCSV } from "./multi_export_utils";
import type { MultiModelResult } from "./types";
import type { CostEstimateResponse } from "../types";

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: vi.fn((v: number, d: number = 0) =>
    Number.isFinite(v) ? v.toFixed(d) : "-"
  ),
}));

function makeCostResponse(overrides: Partial<CostEstimateResponse> = {}): CostEstimateResponse {
  return {
    model: "gpt-4",
    input_tokens: 1000,
    output_tokens: 500,
    num_requests_per_day: 100,
    num_requests_per_month: 3000,
    cost_per_request: 0.05,
    input_cost_per_request: 0.03,
    output_cost_per_request: 0.02,
    margin_cost_per_request: 0,
    daily_cost: 5.0,
    daily_input_cost: 3.0,
    daily_output_cost: 2.0,
    daily_margin_cost: 0,
    monthly_cost: 150.0,
    monthly_input_cost: 90.0,
    monthly_output_cost: 60.0,
    monthly_margin_cost: 0,
    input_cost_per_token: 0.00003,
    output_cost_per_token: 0.00004,
    provider: "openai",
    ...overrides,
  };
}

function makeMultiResult(overrides: Partial<MultiModelResult> = {}): MultiModelResult {
  return {
    entries: [
      {
        entry: { id: "entry-1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 },
        result: makeCostResponse(),
        loading: false,
        error: null,
      },
    ],
    totals: {
      cost_per_request: 0.05,
      daily_cost: 5.0,
      monthly_cost: 150.0,
      margin_per_request: 0,
      daily_margin: null,
      monthly_margin: null,
    },
    ...overrides,
  };
}

describe("exportMultiToPDF", () => {
  let mockPrintWindow: {
    document: { write: ReturnType<typeof vi.fn>; close: ReturnType<typeof vi.fn> };
    print: ReturnType<typeof vi.fn>;
    onload: (() => void) | null;
  };

  beforeEach(() => {
    mockPrintWindow = {
      document: { write: vi.fn(), close: vi.fn() },
      print: vi.fn(),
      onload: null,
    };
    vi.spyOn(window, "open").mockReturnValue(mockPrintWindow as unknown as Window);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should open a new popup window", () => {
    exportMultiToPDF(makeMultiResult());
    expect(window.open).toHaveBeenCalledWith("", "_blank");
  });

  it("should write HTML containing the report title", () => {
    exportMultiToPDF(makeMultiResult());
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).toContain("LLM Cost Estimate Report");
  });

  it("should include model name and provider in the generated HTML", () => {
    exportMultiToPDF(makeMultiResult());
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).toContain("gpt-4");
    expect(html).toContain("openai");
  });

  it("should close the document after writing", () => {
    exportMultiToPDF(makeMultiResult());
    expect(mockPrintWindow.document.close).toHaveBeenCalledTimes(1);
  });

  it("should call print after the window finishes loading", () => {
    exportMultiToPDF(makeMultiResult());
    expect(mockPrintWindow.print).not.toHaveBeenCalled();
    mockPrintWindow.onload!();
    expect(mockPrintWindow.print).toHaveBeenCalledTimes(1);
  });

  it("should show the margin section when margin per request is greater than zero", () => {
    const multiResult = makeMultiResult({
      totals: {
        cost_per_request: 0.06,
        daily_cost: 5.0,
        monthly_cost: 150.0,
        margin_per_request: 0.01,
        daily_margin: 1.0,
        monthly_margin: 30.0,
      },
    });
    exportMultiToPDF(multiResult);
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).toContain("Margin/Request");
  });

  it("should not show the margin section when margin per request is zero", () => {
    exportMultiToPDF(makeMultiResult());
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).not.toContain("Margin/Request");
  });

  it("should alert when popup is blocked", () => {
    vi.spyOn(window, "open").mockReturnValue(null);
    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => {});
    exportMultiToPDF(makeMultiResult());
    expect(alertSpy).toHaveBeenCalledWith("Please allow popups to export PDF");
  });

  it("should only include entries that have a result", () => {
    const multiResult: MultiModelResult = {
      entries: [
        { entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 }, result: null, loading: false, error: null },
        { entry: { id: "e2", model: "claude-3", input_tokens: 500, output_tokens: 250 }, result: makeCostResponse({ model: "claude-3", provider: "anthropic" }), loading: false, error: null },
      ],
      totals: { cost_per_request: 0.05, daily_cost: 5.0, monthly_cost: 150.0, margin_per_request: 0, daily_margin: null, monthly_margin: null },
    };
    exportMultiToPDF(multiResult);
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).toContain("1 model configured");
    expect(html).toContain("claude-3");
  });

  it("should show plural 'models' when multiple results are present", () => {
    const multiResult: MultiModelResult = {
      entries: [
        { entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 }, result: makeCostResponse(), loading: false, error: null },
        { entry: { id: "e2", model: "claude-3", input_tokens: 500, output_tokens: 250 }, result: makeCostResponse({ model: "claude-3" }), loading: false, error: null },
      ],
      totals: { cost_per_request: 0.10, daily_cost: 10.0, monthly_cost: 300.0, margin_per_request: 0, daily_margin: null, monthly_margin: null },
    };
    exportMultiToPDF(multiResult);
    const html = mockPrintWindow.document.write.mock.calls[0][0] as string;
    expect(html).toContain("2 models configured");
  });
});

describe("exportMultiToCSV", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    window.URL.createObjectURL = vi.fn(() => "blob:mock-url");
    window.URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should create an object URL and revoke it after download", () => {
    exportMultiToCSV(makeMultiResult());
    expect(window.URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(window.URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });

  it("should set the download filename to include today's date", () => {
    const createdAnchors: HTMLAnchorElement[] = [];
    const originalCreate = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag: string) => {
      const el = originalCreate(tag);
      if (tag === "a") createdAnchors.push(el as HTMLAnchorElement);
      return el;
    });

    const today = new Date().toISOString().split("T")[0];
    exportMultiToCSV(makeMultiResult());

    expect(createdAnchors[0].download).toBe(`cost_estimate_multi_model_${today}.csv`);
  });

  it("should generate CSV content containing a header row and model data", () => {
    let csvContent = "";
    const OriginalBlob = globalThis.Blob;
    globalThis.Blob = class extends OriginalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        if (typeof parts?.[0] === "string") csvContent = parts[0];
      }
    } as unknown as typeof Blob;

    exportMultiToCSV(makeMultiResult());
    globalThis.Blob = OriginalBlob;

    expect(csvContent).toContain("Model");
    expect(csvContent).toContain("Cost/Request");
    expect(csvContent).toContain("gpt-4");
    expect(csvContent).toContain("openai");
  });

  it("should include the combined totals section in CSV", () => {
    let csvContent = "";
    const OriginalBlob = globalThis.Blob;
    globalThis.Blob = class extends OriginalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        if (typeof parts?.[0] === "string") csvContent = parts[0];
      }
    } as unknown as typeof Blob;

    exportMultiToCSV(makeMultiResult());
    globalThis.Blob = OriginalBlob;

    expect(csvContent).toContain("COMBINED TOTALS");
  });

  it("should create a blob with the correct CSV mime type", () => {
    let capturedType = "";
    const OriginalBlob = globalThis.Blob;
    globalThis.Blob = class extends OriginalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        if (options?.type) capturedType = options.type;
      }
    } as unknown as typeof Blob;

    exportMultiToCSV(makeMultiResult());
    globalThis.Blob = OriginalBlob;

    expect(capturedType).toBe("text/csv;charset=utf-8;");
  });

  it("should skip entries with null results", () => {
    const multiResult: MultiModelResult = {
      entries: [
        { entry: { id: "e1", model: "gpt-4", input_tokens: 1000, output_tokens: 500 }, result: null, loading: false, error: null },
      ],
      totals: { cost_per_request: 0, daily_cost: null, monthly_cost: null, margin_per_request: 0, daily_margin: null, monthly_margin: null },
    };

    let csvContent = "";
    const OriginalBlob = globalThis.Blob;
    globalThis.Blob = class extends OriginalBlob {
      constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
        super(parts, options);
        if (typeof parts?.[0] === "string") csvContent = parts[0];
      }
    } as unknown as typeof Blob;

    exportMultiToCSV(multiResult);
    globalThis.Blob = OriginalBlob;

    // CSV should have metadata rows but no model data row for gpt-4
    const lines = csvContent.split("\n").filter((l) => l.includes('"gpt-4"'));
    expect(lines).toHaveLength(0);
  });
});
