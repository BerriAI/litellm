import React from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import AddPluginForm from "./add_plugin_form";

vi.mock("../networking", () => ({
  registerClaudeCodePlugin: vi.fn().mockResolvedValue({ status: "success" }),
}));

const DEFAULT_PROPS = {
  visible: true,
  onClose: vi.fn(),
  accessToken: "sk-test",
  onSuccess: vi.fn(),
};

describe("AddPluginForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the source type select with GitHub as default", () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    // The default value "GitHub" is displayed in the collapsed select
    expect(screen.getByText("GitHub")).toBeInTheDocument();
    // The form label is present
    expect(screen.getByText("Source Type")).toBeInTheDocument();
  });

  it("shows URL and Path fields when git-subdir is selected", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const sourceSelect = screen.getByLabelText("Source Type");
    await act(async () => {
      fireEvent.mouseDown(sourceSelect);
    });

    await waitFor(() => {
      fireEvent.click(screen.getByText("Git Subdir"));
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("https://github.com/org/repo.git")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("plugins/plugin-name")).toBeInTheDocument();
    });
  });

  it("does not show Path field for url source type", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const sourceSelect = screen.getByLabelText("Source Type");
    await act(async () => {
      fireEvent.mouseDown(sourceSelect);
    });

    await waitFor(() => {
      fireEvent.click(screen.getByText("Git URL"));
    });

    await waitFor(() => {
      expect(screen.getByPlaceholderText("https://github.com/org/repo.git")).toBeInTheDocument();
      expect(screen.queryByPlaceholderText("plugins/plugin-name")).not.toBeInTheDocument();
    });
  });

  it("clears path field when switching away from git-subdir", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    // Switch to git-subdir
    const sourceSelect = screen.getByLabelText("Source Type");
    await act(async () => {
      fireEvent.mouseDown(sourceSelect);
    });
    await waitFor(() => {
      fireEvent.click(screen.getByText("Git Subdir"));
    });

    // Switch back to GitHub
    await act(async () => {
      fireEvent.mouseDown(sourceSelect);
    });
    await waitFor(() => {
      fireEvent.click(screen.getByText("GitHub"));
    });

    await waitFor(() => {
      expect(screen.queryByPlaceholderText("plugins/plugin-name")).not.toBeInTheDocument();
      expect(screen.getByPlaceholderText("anthropics/claude-code")).toBeInTheDocument();
    });
  });
});
