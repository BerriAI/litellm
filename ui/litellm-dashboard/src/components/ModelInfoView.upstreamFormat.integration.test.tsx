import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ModelInfoView from "./model_info_view";
import { modelPatchUpdateCall } from "./networking";

const mockUseModelsInfo = vi.fn();

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: () => mockUseModelsInfo(),
  useModelHub: () => ({ data: { data: [] } }),
}));
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: {} }),
}));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => false,
}));

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
    vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
    credentialGetCall: vi.fn().mockResolvedValue({ credential_name: "", credential_values: {}, credential_info: {} }),
    credentialListCall: vi.fn().mockResolvedValue({ credentials: [] }),
    getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
    tagListCall: vi.fn().mockResolvedValue({}),
  };
});

const renderModel = async (model: string) => {
  const modelData = {
    model_name: "custom-model",
    litellm_params: { model, custom_llm_provider: "openai" },
    model_info: { id: "model-id", db_model: true, mode: "chat", created_by: "admin" },
  };
  mockUseModelsInfo.mockReturnValue({ data: { data: [modelData] }, isLoading: false });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <ModelInfoView
        modelId="model-id"
        accessToken="test-token"
        userID="admin"
        userRole="Admin"
        isViewOnly={false}
        modelAccessGroups={[]}
        onClose={vi.fn()}
      />
    </QueryClientProvider>,
  );
  const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
  return user;
};

const savedModel = () => vi.mocked(modelPatchUpdateCall).mock.calls[0][1];

describe("ModelInfoView upstream format", () => {
  beforeEach(() => vi.clearAllMocks());

  it("should render the saved format with a readable model name", async () => {
    await renderModel("openai/systemone/org/model");

    expect(screen.getByRole("combobox", { name: "Upstream format" })).toBeInTheDocument();
    expect(screen.getByText("System One")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter LiteLLM model name")).toHaveValue("openai/org/model");
  });

  it.each([
    "openai/org/model",
    "openai/responses/org/model",
    "openai/systemone/org/model",
    "openai/decisions/org/model",
    "openai/systemone/systemone/model",
    "openai/decisions/decisions/model",
    "openai/systemone/decisions/model",
  ])("should preserve %s when saving without changing the format", async (model) => {
    const user = await renderModel(model);
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedModel().litellm_params.model).toBe(model);
    expect(savedModel().litellm_params).not.toHaveProperty("upstream_format");
    expect(savedModel().model_info.mode).toBe("chat");
  });

  it("should save the chosen format and the edited upstream model name", async () => {
    const user = await renderModel("openai/systemone/org/model");
    await user.click(screen.getByRole("combobox", { name: "Upstream format" }));
    await user.click(await screen.findByRole("option", { name: "OpenAI Decisions" }));
    act(() =>
      fireEvent.change(screen.getByPlaceholderText("Enter LiteLLM model name"), {
        target: { value: "openai/org/updated/model" },
      }),
    );
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedModel().litellm_params.model).toBe("openai/decisions/org/updated/model");
    expect(await screen.findByText("openai/decisions/org/updated/model")).toBeInTheDocument();
  });

  it("should remove a decision format when the provider default is selected", async () => {
    const user = await renderModel("openai/decisions/org/model");
    await user.click(screen.getByRole("combobox", { name: "Upstream format" }));
    await user.click(await screen.findByRole("option", { name: "Default" }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedModel().litellm_params.model).toBe("openai/org/model");
  });

  it("should stop applying the OpenAI format when the provider changes", async () => {
    const user = await renderModel("openai/systemone/org/model");
    act(() => {
      fireEvent.change(screen.getByPlaceholderText("Enter custom LLM provider"), { target: { value: "anthropic" } });
      fireEvent.change(screen.getByPlaceholderText("Enter LiteLLM model name"), {
        target: { value: "anthropic/custom-model" },
      });
    });
    expect(screen.queryByRole("combobox", { name: "Upstream format" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalledTimes(1));
    expect(savedModel().litellm_params).toMatchObject({
      model: "anthropic/custom-model",
      custom_llm_provider: "anthropic",
    });
  });
});
