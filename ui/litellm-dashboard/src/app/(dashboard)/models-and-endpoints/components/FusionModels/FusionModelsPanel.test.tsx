/* @vitest-environment jsdom */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FusionModelsPanel } from "./FusionModelsPanel";

const modelCreateCall = vi.fn();
const modelPatchUpdateCall = vi.fn();
const modelDeleteCall = vi.fn();
const invalidate = vi.fn().mockResolvedValue(undefined);
const useFusionRouters = vi.fn();

vi.mock("@/components/networking", () => ({
  modelCreateCall: (...args: unknown[]) => modelCreateCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => modelPatchUpdateCall(...args),
  modelDeleteCall: (...args: unknown[]) => modelDeleteCall(...args),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useFusionRouters: () => useFusionRouters(),
  useInvalidateFusionRouters: () => invalidate,
  usePlainModelGroups: () => new Set(["panel-a", "panel-b", "aggregator"]),
}));

vi.mock("@/components/shared/MultiSelect", () => ({
  MultiSelect: ({ onValueChange }: { onValueChange: (models: string[]) => void }) => (
    <button type="button" onClick={() => onValueChange(["panel-a", "panel-b"])}>
      Choose panel models
    </button>
  ),
}));

vi.mock("@/components/common_components/team_dropdown", () => ({
  default: ({ onChange }: { onChange: (teamID: string) => void }) => (
    <button type="button" onClick={() => onChange("team-1")}>
      Choose team
    </button>
  ),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({ default: () => null }));
vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

const existingDeployment = {
  model_name: "fusion/existing",
  litellm_params: {
    model: "fusion_router",
    fusion_router_config: {
      panel_models: ["panel-a", "panel-b"],
      aggregator_model: "aggregator",
      min_successful_panelists: 2,
      panel_timeout_seconds: 120,
      max_candidate_chars: 12000,
      on_quorum_failure: "aggregator_only",
    },
  },
  model_info: { id: "fusion-id", db_model: true },
};

const renderPanel = (createScope: "unscoped-ok" | "team-required" = "unscoped-ok") =>
  render(
    <FusionModelsPanel accessToken="token" userRole="Admin" userID="user-1" teams={[]} createScope={createScope} />,
  );

describe("FusionModelsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFusionRouters.mockReturnValue({ data: [], isLoading: false });
    modelCreateCall.mockResolvedValue({});
    modelPatchUpdateCall.mockResolvedValue({});
  });

  it("creates a quality-first Fusion model with an ordinary model/new payload", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    await user.type(screen.getByLabelText("Model name"), "fusion/coding");
    await user.click(screen.getByRole("button", { name: "Choose panel models" }));
    await user.click(screen.getByLabelText("Aggregator model"));
    await user.click(screen.getByRole("option", { name: "aggregator" }));
    await user.click(screen.getByRole("button", { name: "Create Fusion Model" }));

    await waitFor(() => expect(modelCreateCall).toHaveBeenCalledTimes(1));
    expect(modelCreateCall).toHaveBeenCalledWith("token", {
      model_name: "fusion/coding",
      litellm_params: {
        model: "fusion_router",
        fusion_router_config: {
          panel_models: ["panel-a", "panel-b"],
          aggregator_model: "aggregator",
          min_successful_panelists: 2,
          panel_timeout_seconds: 120,
          max_candidate_chars: 12000,
          on_quorum_failure: "fail",
        },
      },
      model_info: {},
    });
  });

  it("requires and sends a team for team-admin creation", async () => {
    const user = userEvent.setup();
    renderPanel("team-required");
    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    await user.type(screen.getByLabelText("Model name"), "fusion/team");
    await user.click(screen.getByRole("button", { name: "Choose panel models" }));
    await user.click(screen.getByLabelText("Aggregator model"));
    await user.click(screen.getByRole("option", { name: "aggregator" }));
    await user.click(screen.getByRole("button", { name: "Create Fusion Model" }));
    expect(await screen.findByText("Select a team to continue.")).toBeInTheDocument();
    expect(modelCreateCall).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Choose team" }));
    await user.click(screen.getByRole("button", { name: "Create Fusion Model" }));
    await waitFor(() => expect(modelCreateCall).toHaveBeenCalled());
    expect(modelCreateCall.mock.calls[0][1].model_info).toEqual({ team_id: "team-1" });
  });

  it("edits the stored Fusion config without renaming the public model", async () => {
    useFusionRouters.mockReturnValue({ data: [existingDeployment], isLoading: false });
    const user = userEvent.setup();
    renderPanel();

    expect(screen.getByText("High Availability")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Configure fusion/existing" }));
    expect(screen.getByLabelText("Model name")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(modelPatchUpdateCall).toHaveBeenCalledWith(
      "token",
      {
        litellm_params: expect.objectContaining({
          model: "fusion_router",
          fusion_router_config: expect.objectContaining({ on_quorum_failure: "aggregator_only" }),
        }),
      },
      "fusion-id",
    );
  });
});
