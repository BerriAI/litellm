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
const toastSuccess = vi.fn();
const toastFromError = vi.fn();

vi.mock("@/components/networking", () => ({
  modelCreateCall: (...args: unknown[]) => modelCreateCall(...args),
  modelPatchUpdateCall: (...args: unknown[]) => modelPatchUpdateCall(...args),
  modelDeleteCall: (...args: unknown[]) => modelDeleteCall(...args),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useFusionRouters: () => useFusionRouters(),
  useInvalidateFusionRouters: () => invalidate,
  usePlainModelGroups: () => new Set(["panel-a", "panel-b", "outer", "analyst"]),
}));

vi.mock("@/components/shared/MultiSelect", () => ({
  MultiSelect: ({ onValueChange }: { onValueChange: (models: string[]) => void }) => (
    <button type="button" onClick={() => onValueChange(["panel-a", "panel-b"])}>
      Choose panel models
    </button>
  ),
}));

vi.mock("@/components/search_tools/SearchToolSelector", () => ({
  default: ({
    onChange,
    onOptionsLoaded,
  }: {
    onChange: (tools: string[]) => void;
    onOptionsLoaded?: (tools: string[]) => void;
  }) => (
    <div data-testid="search-tool-selector">
      <button type="button" onClick={() => onOptionsLoaded?.(["web-search"])}>
        Load search tools
      </button>
      <button type="button" onClick={() => onChange(["web-search"])}>
        Select web search
      </button>
    </div>
  ),
}));

vi.mock("@/components/common_components/team_dropdown", () => ({
  default: ({ onChange }: { onChange: (teamID: string) => void }) => (
    <button type="button" onClick={() => onChange("team-1")}>
      Choose team
    </button>
  ),
}));

vi.mock("@/components/common_components/DeleteResourceModal", () => ({
  default: ({ title, message, onOk }: { title: string; message: string; onOk: () => void }) => (
    <div role="dialog" aria-label={title}>
      <p>{message}</p>
      <button type="button" onClick={onOk}>
        Confirm delete
      </button>
    </div>
  ),
}));
vi.mock("@/lib/toast", () => ({
  toast: {
    success: (...args: unknown[]) => toastSuccess(...args),
    fromError: (...args: unknown[]) => toastFromError(...args),
  },
}));

const existingDeployment = {
  model_name: "fusion/existing",
  litellm_params: {
    model: "fusion_router",
    fusion_router_config: {
      outer_model: "outer",
      panel_models: ["panel-a", "panel-b"],
      analyst_model: "analyst",
      invocation: "required",
      panel_timeout_seconds: 120,
      max_candidate_chars: 12000,
      max_completion_tokens: 16000,
      temperature: 0,
      reasoning_effort: "none",
      max_tool_calls: 4,
    },
  },
  model_info: { id: "fusion-id", db_model: true },
};

const renderPanel = (createScope: "unscoped-ok" | "team-required" = "unscoped-ok") =>
  render(
    <FusionModelsPanel
      accessToken="token"
      userRole="Admin"
      userID="user-1"
      isViewOnly={false}
      teams={[]}
      createScope={createScope}
    />,
  );

describe("FusionModelsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFusionRouters.mockReturnValue({ data: [], isLoading: false });
    modelCreateCall.mockResolvedValue({});
    modelPatchUpdateCall.mockResolvedValue({});
    modelDeleteCall.mockResolvedValue({});
  });

  it("creates an auto Fusion model with an ordinary model/new payload", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    await user.type(screen.getByLabelText("Model name"), "fusion/coding");
    await user.click(screen.getByRole("button", { name: "Choose panel models" }));
    await user.click(screen.getByLabelText("Outer model"));
    await user.click(await screen.findByRole("option", { name: "outer" }, { timeout: 5000 }));
    await user.click(screen.getByRole("button", { name: "Load search tools" }));
    await user.click(screen.getByRole("button", { name: "Create Fusion Model" }));

    await waitFor(() => expect(modelCreateCall).toHaveBeenCalledTimes(1));
    expect(modelCreateCall).toHaveBeenCalledWith("token", {
      model_name: "fusion/coding",
      litellm_params: {
        model: "fusion_router",
        fusion_router_config: {
          outer_model: "outer",
          panel_models: ["panel-a", "panel-b"],
          invocation: "auto",
          panel_timeout_seconds: 120,
          max_candidate_chars: 12000,
          max_completion_tokens: 16000,
          temperature: 0,
          reasoning_effort: "none",
          search_tool_name: "web-search",
          max_tool_calls: 4,
        },
      },
      model_info: {},
    });
  });

  it("keeps the form open and shows the backend error when creation fails", async () => {
    modelCreateCall.mockRejectedValue(new Error("backend unavailable"));
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    await user.type(screen.getByLabelText("Model name"), "fusion/coding");
    await user.click(screen.getByRole("button", { name: "Choose panel models" }));
    await user.click(screen.getByLabelText("Outer model"));
    await user.click(await screen.findByRole("option", { name: "outer" }, { timeout: 5000 }));
    await user.click(screen.getByRole("button", { name: "Load search tools" }));
    await user.click(screen.getByRole("button", { name: "Create Fusion Model" }));

    expect(await screen.findByText("backend unavailable")).toBeVisible();
    expect(screen.getByRole("dialog", { name: "Add Fusion Model" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Create Fusion Model" })).toBeEnabled();
    expect(invalidate).not.toHaveBeenCalled();
  });

  it("shows readable, full-width deliberation presets", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    const behaviorSelector = screen.getByLabelText("Behavior");
    expect(behaviorSelector).toHaveTextContent("Auto");
    expect(behaviorSelector).toHaveClass("w-full");

    await user.click(behaviorSelector);
    expect(await screen.findByRole("option", { name: /^Auto/ }, { timeout: 5000 })).toBeVisible();
    expect(await screen.findByRole("option", { name: /Always deliberate/ }, { timeout: 5000 })).toBeVisible();

    await user.click(await screen.findByRole("option", { name: /Always deliberate/ }, { timeout: 5000 }));
    expect(behaviorSelector).toHaveTextContent("Always deliberate");

    expect(screen.getByRole("switch", { name: "Web access" })).toBeChecked();
    expect(screen.getByTestId("search-tool-selector")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Advanced settings" }));
    expect(screen.getByLabelText("Maximum searches per internal model")).toHaveValue(4);
  });

  it("requires and sends a team for team-admin creation", async () => {
    const user = userEvent.setup();
    renderPanel("team-required");
    await user.click(screen.getByRole("button", { name: "Add Fusion Model" }));
    await user.type(screen.getByLabelText("Model name"), "fusion/team");
    await user.click(screen.getByRole("button", { name: "Choose panel models" }));
    await user.click(screen.getByLabelText("Outer model"));
    await user.click(await screen.findByRole("option", { name: "outer" }, { timeout: 5000 }));
    await user.click(screen.getByRole("button", { name: "Load search tools" }));
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

    expect(screen.getByText("Always")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Configure fusion/existing" }));
    expect(screen.getByLabelText("Model name")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(modelPatchUpdateCall).toHaveBeenCalledWith(
      "token",
      {
        litellm_params: expect.objectContaining({
          model: "fusion_router",
          fusion_router_config: expect.objectContaining({ outer_model: "outer", invocation: "required" }),
        }),
      },
      "fusion-id",
    );
  });

  it("deletes a stored Fusion model and refreshes the list", async () => {
    useFusionRouters.mockReturnValue({ data: [existingDeployment], isLoading: false });
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Delete fusion/existing" }));
    expect(screen.getByRole("dialog", { name: "Delete Fusion Model" })).toHaveTextContent("fusion/existing");
    await user.click(screen.getByRole("button", { name: "Confirm delete" }));

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("token", "fusion-id"));
    await waitFor(() => expect(invalidate).toHaveBeenCalled());
    expect(toastSuccess).toHaveBeenCalledWith("Deleted Fusion model: fusion/existing");
  });

  it("keeps the delete dialog open when deletion fails", async () => {
    useFusionRouters.mockReturnValue({ data: [existingDeployment], isLoading: false });
    modelDeleteCall.mockRejectedValue(new Error("backend unavailable"));
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("button", { name: "Delete fusion/existing" }));
    await user.click(screen.getByRole("button", { name: "Confirm delete" }));

    await waitFor(() => expect(toastFromError).toHaveBeenCalled());
    expect(screen.getByRole("dialog", { name: "Delete Fusion Model" })).toBeVisible();
    expect(invalidate).not.toHaveBeenCalled();
  });
});
