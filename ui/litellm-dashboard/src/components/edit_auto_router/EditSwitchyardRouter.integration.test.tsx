import React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import EditSwitchyardRouter from "./EditSwitchyardRouter";

const { patch } = vi.hoisted(() => ({ patch: vi.fn().mockResolvedValue({}) }));
vi.mock("../networking", () => ({ modelPatchUpdateCall: patch }));
vi.mock("../llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
  fetchAutoRouterModels: vi.fn().mockResolvedValue([]),
}));
const modelData = {
  model_name: "switchyard",
  model_info: { id: "router-id" },
  litellm_params: {
    model: "auto_router/switchyard_router",
    switchyard_router_config: {
      algorithm: "capability",
      models: { efficient: ["small"], capable: ["large"], judge: ["judge"] },
      capability: { base_threshold: 0.7, future_option: "preserve" },
    },
  },
};

beforeEach(() => vi.clearAllMocks());
describe("Switchyard editor", () => {
  it("saves advanced edits under the strategy key and preserves unrecognized settings", async () => {
    const onSuccess = vi.fn();
    renderWithProviders(
      <EditSwitchyardRouter
        isVisible
        accessToken="test"
        modelData={modelData}
        onCancel={vi.fn()}
        onSuccess={onSuccess}
      />,
    );
    fireEvent.click(screen.getByText("Advanced settings"));
    fireEvent.change(screen.getByLabelText("Capability threshold"), { target: { value: "0.8" } });
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() =>
      expect(patch).toHaveBeenCalledWith(
        "test",
        {
          model_name: "switchyard",
          litellm_params: {
            switchyard_router_config: {
              ...modelData.litellm_params.switchyard_router_config,
              capability: { base_threshold: 0.8, future_option: "preserve" },
            },
          },
        },
        "router-id",
      ),
    );
    expect(onSuccess).toHaveBeenCalled();
  });

  it("blocks invalid tuning and permits restoring defaults before save", () => {
    renderWithProviders(
      <EditSwitchyardRouter
        isVisible
        accessToken="test"
        modelData={modelData}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByText("Advanced settings"));
    fireEvent.change(screen.getByLabelText("Capability threshold"), { target: { value: "2" } });
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Restore defaults" }));
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeEnabled();
  });

  it("blocks unsupported member-managed updates", () => {
    renderWithProviders(
      <EditSwitchyardRouter
        isVisible
        isMemberManaged
        accessToken="test"
        modelData={modelData}
        onCancel={vi.fn()}
        onSuccess={vi.fn()}
      />,
    );
    expect(screen.getByLabelText("Auto Router Name")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("must be edited by a team or proxy administrator");
    expect(patch).not.toHaveBeenCalled();
  });
});
