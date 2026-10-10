/* @vitest-environment jsdom */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { setCallbacksCall } from "@/components/networking";
import ModelGroupAliasSettings from "./model_group_alias_settings";

vi.mock("@/components/networking", () => ({
  setCallbacksCall: vi.fn(),
}));

const aliases = {
  "fast-model": "claude-haiku",
  "smart-model": { model: "claude-sonnet", hidden: true },
};

describe("ModelGroupAliasSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(setCallbacksCall).mockResolvedValue({});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("locks config-owned aliases while keeping their names and targets visible", async () => {
    render(<ModelGroupAliasSettings accessToken="test-token" initialModelGroupAlias={aliases} managedByConfig />);

    expect(
      screen.getByText(
        "These aliases are defined in config.yaml and are read-only here. Edit the config file to change them.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., gpt-4o")).toBeDisabled();
    expect(screen.getByPlaceholderText("e.g., gpt-4o-mini-openai")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add Alias" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit alias fast-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit alias smart-model" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete alias smart-model" })).toBeDisabled();

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByText("claude-haiku")).toBeInTheDocument();
    expect(screen.getByText("smart-model")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet")).toBeInTheDocument();

    await userEvent.setup().click(screen.getByRole("button", { name: "Delete alias fast-model" }));
    expect(setCallbacksCall).not.toHaveBeenCalled();
  });

  it("keeps aliases editable by default and saves the current string representation after delete", async () => {
    const user = userEvent.setup();
    render(<ModelGroupAliasSettings accessToken="test-token" initialModelGroupAlias={aliases} />);

    expect(await screen.findByText("fast-model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit alias fast-model" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete alias fast-model" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Delete alias fast-model" }));

    expect(setCallbacksCall).toHaveBeenCalledWith("test-token", {
      router_settings: { model_group_alias: { "smart-model": "claude-sonnet" } },
    });
  });
});
