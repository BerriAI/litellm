import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import ContentFilterManager from "./ContentFilterManager";
import React from "react";

vi.mock("./ContentFilterConfiguration", () => ({
  default: () => <div data-testid="content-filter-config">Mock Content Filter Configuration</div>
}));

vi.mock("./ContentFilterDisplay", () => ({
  default: () => <div data-testid="content-filter-display">Mock Content Filter Display</div>
}));

vi.mock("antd", () => ({
  Divider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}));

describe("ContentFilterManager - Unsaved Changes Detection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call onUnsavedChanges with false when component initializes with matching data", async () => {
    /**
     * Tests that the ContentFilterManager correctly initializes the unsaved changes
     * detection and calls onUnsavedChanges(false) when the current state matches
     * the original loaded state (no changes yet).
     */
    const mockOnUnsavedChanges = vi.fn();
    const mockOnDataChange = vi.fn();

    const guardrailData = {
      litellm_params: {
        guardrail: "litellm_content_filter",
        patterns: [
          { pattern_type: "prebuilt", pattern_name: "email", action: "BLOCK" }
        ],
        blocked_words: [
          { keyword: "test", action: "BLOCK", description: null }
        ]
      }
    };

    const guardrailSettings = {
      content_filter_settings: {
        prebuilt_patterns: [],
        pattern_categories: ["PII"],
        supported_actions: ["BLOCK", "MASK"]
      }
    };

    render(
      <ContentFilterManager
        guardrailData={guardrailData}
        guardrailSettings={guardrailSettings}
        isEditing={true}
        accessToken="test-token"
        onDataChange={mockOnDataChange}
        onUnsavedChanges={mockOnUnsavedChanges}
      />
    );

    // Wait for component to render in edit mode
    await waitFor(() => {
      expect(screen.getByTestId("content-filter-config")).toBeInTheDocument();
    });

    // Verify onUnsavedChanges was called with false (no changes initially)
    await waitFor(() => {
      expect(mockOnUnsavedChanges).toHaveBeenCalledWith(false);
    });

    // Verify onDataChange was called with initial data
    expect(mockOnDataChange).toHaveBeenCalled();
  });
});

