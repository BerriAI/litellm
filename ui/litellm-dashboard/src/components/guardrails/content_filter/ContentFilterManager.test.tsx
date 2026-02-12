import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ContentFilterManager, {
  formatContentFilterDataForAPI,
} from "./ContentFilterManager";
import React from "react";

const CONTENT_FILTER_GUARDRAIL_DATA = {
  litellm_params: {
    guardrail: "litellm_content_filter",
    patterns: [{ pattern_type: "prebuilt", pattern_name: "email", action: "BLOCK" }],
    blocked_words: [{ keyword: "test", action: "BLOCK", description: null }],
  },
};

const GUARDRAIL_SETTINGS = {
  content_filter_settings: {
    prebuilt_patterns: [],
    pattern_categories: ["PII"],
    supported_actions: ["BLOCK", "MASK"],
  },
};

vi.mock("./ContentFilterConfiguration", () => ({
  default: ({
    onPatternAdd,
    onPatternRemove,
    onBlockedWordAdd,
    onBlockedWordRemove,
    selectedPatterns,
    blockedWords,
  }: {
    onPatternAdd: (p: object) => void;
    onPatternRemove: (id: string) => void;
    onBlockedWordAdd: (w: object) => void;
    onBlockedWordRemove: (id: string) => void;
    selectedPatterns: { id: string }[];
    blockedWords: { id: string }[];
  }) => (
    <div data-testid="content-filter-config">
      <button
        type="button"
        onClick={() =>
          onPatternAdd({
            id: "pattern-new",
            type: "prebuilt",
            name: "ssn",
            action: "BLOCK",
          })
        }
      >
        Add pattern
      </button>
      <button
        type="button"
        onClick={() =>
          onBlockedWordAdd({
            id: "word-new",
            keyword: "secret",
            action: "MASK",
          })
        }
      >
        Add keyword
      </button>
      {selectedPatterns[0] && (
        <button
          type="button"
          onClick={() => onPatternRemove(selectedPatterns[0].id)}
        >
          Remove pattern
        </button>
      )}
      {blockedWords[0] && (
        <button
          type="button"
          onClick={() => onBlockedWordRemove(blockedWords[0].id)}
        >
          Remove keyword
        </button>
      )}
    </div>
  ),
}));

vi.mock("./ContentFilterDisplay", () => ({
  default: ({
    patterns,
    blockedWords,
  }: {
    patterns: { name: string }[];
    blockedWords: { keyword: string }[];
  }) => (
    <div data-testid="content-filter-display">
      <span>Patterns: {patterns.map((p) => p.name).join(", ")}</span>
      <span>Keywords: {blockedWords.map((w) => w.keyword).join(", ")}</span>
    </div>
  ),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    Divider: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="divider">{children}</div>
    ),
    Alert: ({
      message,
      type,
    }: {
      message: React.ReactNode;
      type: string;
    }) => (
      <div data-testid="unsaved-alert" data-type={type}>
        {message}
      </div>
    ),
  };
});

describe("ContentFilterManager", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render when guardrail is content filter and isEditing is true", async () => {
    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    expect(screen.getByTestId("divider")).toHaveTextContent(
      "Content Filter Configuration"
    );
  });

  it("should return null when guardrail is not litellm_content_filter", () => {
    const guardrailData = {
      litellm_params: { guardrail: "presidio" },
    };

    const { container } = render(
      <ContentFilterManager
        guardrailData={guardrailData}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
      />
    );

    expect(screen.queryByTestId("content-filter-config")).not.toBeInTheDocument();
    expect(screen.queryByTestId("content-filter-display")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  it("should render read-only display when isEditing is false", async () => {
    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={false}
        accessToken="test-token"
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-display")).toBeInTheDocument();
    });

    expect(screen.queryByTestId("content-filter-config")).not.toBeInTheDocument();
    expect(screen.getByText(/email/)).toBeInTheDocument();
    expect(screen.getByText(/test/)).toBeInTheDocument();
  });

  it("should call onUnsavedChanges with false when component initializes with matching data", async () => {
    const mockOnUnsavedChanges = vi.fn();
    const mockOnDataChange = vi.fn();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onDataChange={mockOnDataChange}
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(mockOnUnsavedChanges).toHaveBeenCalledWith(false);
    });

    expect(mockOnDataChange).toHaveBeenCalled();
  });

  it("should call onUnsavedChanges with true when user adds a pattern", async () => {
    const mockOnUnsavedChanges = vi.fn();
    const user = userEvent.setup();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /add pattern/i }));

    await waitFor(() => {
      expect(mockOnUnsavedChanges).toHaveBeenCalledWith(true);
    });
  });

  it("should call onUnsavedChanges with true when user adds a keyword", async () => {
    const mockOnUnsavedChanges = vi.fn();
    const user = userEvent.setup();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /add keyword/i }));

    await waitFor(() => {
      expect(mockOnUnsavedChanges).toHaveBeenCalledWith(true);
    });
  });

  it("should call onUnsavedChanges with true when user removes a pattern", async () => {
    const mockOnUnsavedChanges = vi.fn();
    const user = userEvent.setup();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /remove pattern/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /remove pattern/i }));

    await waitFor(() => {
      expect(mockOnUnsavedChanges).toHaveBeenCalledWith(true);
    });
  });

  it("should show unsaved changes alert when data has changed", async () => {
    const user = userEvent.setup();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    expect(screen.queryByTestId("unsaved-alert")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add pattern/i }));

    await waitFor(() => {
      expect(screen.getByTestId("unsaved-alert")).toBeInTheDocument();
    });

    expect(screen.getByTestId("unsaved-alert")).toHaveTextContent(
      /unsaved changes.*Save Changes/i
    );
  });

  it("should call onDataChange when patterns or keywords change", async () => {
    const mockOnDataChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onDataChange={mockOnDataChange}
      />
    );

    await waitFor(() => {
      expect(mockOnDataChange).toHaveBeenCalled();
    });

    const initialCalls = mockOnDataChange.mock.calls.length;
    await user.click(screen.getByRole("button", { name: /add keyword/i }));

    await waitFor(() => {
      expect(mockOnDataChange.mock.calls.length).toBeGreaterThan(initialCalls);
    });

    const lastCall = mockOnDataChange.mock.calls[mockOnDataChange.mock.calls.length - 1];
    const blockedWords = lastCall[1];
    expect(blockedWords).toContainEqual(
      expect.objectContaining({ keyword: "secret", action: "MASK" })
    );
  });

  it("should not call onUnsavedChanges when isEditing is false", async () => {
    const mockOnUnsavedChanges = vi.fn();

    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={false}
        accessToken="test-token"
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("content-filter-display")).toBeInTheDocument();
    });

    expect(mockOnUnsavedChanges).not.toHaveBeenCalled();
  });

  it("should initialize with empty data when guardrailData has no patterns or blocked_words", async () => {
    const guardrailData = {
      litellm_params: {
        guardrail: "litellm_content_filter",
      },
    };

    const mockOnDataChange = vi.fn();

    render(
      <ContentFilterManager
        guardrailData={guardrailData}
        guardrailSettings={GUARDRAIL_SETTINGS}
        isEditing={true}
        accessToken="test-token"
        onDataChange={mockOnDataChange}
      />
    );

    await waitFor(() => {
      expect(mockOnDataChange).toHaveBeenCalledWith([], [], []);
    });
  });

  it("should not render ContentFilterConfiguration when guardrailSettings has no content_filter_settings", async () => {
    render(
      <ContentFilterManager
        guardrailData={CONTENT_FILTER_GUARDRAIL_DATA}
        guardrailSettings={null}
        isEditing={true}
        accessToken="test-token"
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId("divider")).toBeInTheDocument();
    });

    expect(screen.queryByTestId("content-filter-config")).not.toBeInTheDocument();
  });
});

describe("formatContentFilterDataForAPI", () => {
  it("should format patterns and blocked words for API", () => {
    const patterns = [
      {
        id: "p1",
        type: "prebuilt" as const,
        name: "email",
        action: "BLOCK" as const,
      },
      {
        id: "p2",
        type: "custom" as const,
        name: "custom",
        pattern: "\\d+",
        action: "MASK" as const,
      },
    ];
    const blockedWords = [
      {
        id: "w1",
        keyword: "secret",
        action: "MASK" as const,
        description: "-sensitive",
      },
    ];

    const result = formatContentFilterDataForAPI(patterns, blockedWords);

    expect(result.patterns).toEqual([
      {
        pattern_type: "prebuilt",
        pattern_name: "email",
        pattern: undefined,
        name: "email",
        action: "BLOCK",
      },
      {
        pattern_type: "regex",
        pattern_name: undefined,
        pattern: "\\d+",
        name: "custom",
        action: "MASK",
      },
    ]);
    expect(result.blocked_words).toEqual([
      { keyword: "secret", action: "MASK", description: "-sensitive" },
    ]);
    expect(result.categories).toBeUndefined();
  });

  it("should include categories when provided", () => {
    const patterns: Parameters<typeof formatContentFilterDataForAPI>[0] = [];
    const blockedWords: Parameters<typeof formatContentFilterDataForAPI>[1] = [];
    const categories = [
      {
        id: "c1",
        category: "PII",
        display_name: "PII",
        action: "BLOCK" as const,
        severity_threshold: "high" as const,
      },
    ];

    const result = formatContentFilterDataForAPI(patterns, blockedWords, categories);

    expect(result.categories).toEqual([
      {
        category: "PII",
        enabled: true,
        action: "BLOCK",
        severity_threshold: "high",
      },
    ]);
  });

  it("should use medium as default severity_threshold when category has none", () => {
    const patterns: Parameters<typeof formatContentFilterDataForAPI>[0] = [];
    const blockedWords: Parameters<typeof formatContentFilterDataForAPI>[1] = [];
    const categories = [
      {
        id: "c1",
        category: "PII",
        display_name: "PII",
        action: "MASK" as const,
        severity_threshold: undefined as unknown as "high" | "medium" | "low",
      },
    ];

    const result = formatContentFilterDataForAPI(patterns, blockedWords, categories);

    expect(result.categories).toEqual([
      {
        category: "PII",
        enabled: true,
        action: "MASK",
        severity_threshold: "medium",
      },
    ]);
  });
});
