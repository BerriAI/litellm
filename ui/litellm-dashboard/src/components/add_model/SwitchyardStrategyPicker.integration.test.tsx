import React from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen, waitFor, chooseSelectOption } from "../../../tests/test-utils";
import SwitchyardStrategyPicker from "./SwitchyardStrategyPicker";

const { create } = vi.hoisted(() => ({ create: vi.fn().mockResolvedValue({}) }));
vi.mock("../networking", () => ({ modelCreateCall: create }));
vi.mock("../llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
  fetchAutoRouterModels: vi.fn().mockResolvedValue([]),
}));

const props = { handleOk: vi.fn(), accessToken: "test", userRole: "Admin" };

describe("Switchyard creation", () => {
  it("creates the Switchyard strategy with basic models and advanced tuning", async () => {
    renderWithProviders(
      <SwitchyardStrategyPicker {...props}>
        <p>Complexity setup</p>
      </SwitchyardStrategyPicker>,
    );
    await chooseSelectOption(userEvent.setup(), screen.getByLabelText("Routing strategy"), "Switchyard");
    fireEvent.change(screen.getByLabelText("efficient models"), { target: { value: "small" } });
    fireEvent.change(screen.getByLabelText("capable models"), { target: { value: "large, backup" } });
    fireEvent.change(screen.getByLabelText("judge models"), { target: { value: "judge" } });
    fireEvent.click(screen.getByText("Advanced settings"));
    fireEvent.change(screen.getByLabelText("Capability threshold"), { target: { value: "0.6" } });
    fireEvent.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await waitFor(() =>
      expect(create).toHaveBeenCalledWith("test", {
        model_name: "switchyard-router",
        litellm_params: {
          model: "auto_router/switchyard_router",
          switchyard_router_config: {
            algorithm: "capability",
            models: { efficient: ["small"], capable: ["large", "backup"], judge: ["judge"] },
            fallback_enabled: true,
            capability: { base_threshold: 0.6 },
          },
        },
        model_info: {},
      }),
    );
  });
});
