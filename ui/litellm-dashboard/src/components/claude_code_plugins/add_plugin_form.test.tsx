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

  it("renders with GitHub URL input", () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    expect(screen.getByText("GitHub URL")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText("https://github.com/org/repo/tree/main/my-skill")
    ).toBeInTheDocument();
  });

  it("shows GitHub repo preview for a plain repo URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(
      "https://github.com/org/repo/tree/main/my-skill"
    );

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/claude-code" },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/GitHub repo/)).toBeInTheDocument();
    });
  });

  it("shows git-subdir preview for a tree URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(
      "https://github.com/org/repo/tree/main/my-skill"
    );

    await act(async () => {
      fireEvent.change(urlInput, {
        target: {
          value: "https://github.com/anthropics/claude-code/tree/main/plugins/my-skill",
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText(/GitHub subdir/)).toBeInTheDocument();
    });
  });

  it("auto-fills skill name from repo URL", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const urlInput = screen.getByPlaceholderText(
      "https://github.com/org/repo/tree/main/my-skill"
    );

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/my-awesome-skill" },
      });
    });

    await waitFor(() => {
      const nameInput = screen.getByPlaceholderText("my-skill") as HTMLInputElement;
      expect(nameInput.value).toBe("my-awesome-skill");
    });
  });

  it("does not auto-fill name when name is already set", async () => {
    renderWithProviders(<AddPluginForm {...DEFAULT_PROPS} />);

    const nameInput = screen.getByPlaceholderText("my-skill") as HTMLInputElement;
    fireEvent.change(nameInput, { target: { value: "existing-name" } });

    const urlInput = screen.getByPlaceholderText(
      "https://github.com/org/repo/tree/main/my-skill"
    );

    await act(async () => {
      fireEvent.change(urlInput, {
        target: { value: "https://github.com/anthropics/other-skill" },
      });
    });

    await waitFor(() => {
      expect(nameInput.value).toBe("existing-name");
    });
  });
});
