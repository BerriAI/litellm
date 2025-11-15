import { render, renderHook } from "@testing-library/react";
import { describe, it, vi, expect } from "vitest";
import { Form } from "antd";
import AddModelTab from "./add_model_tab";
import { Providers } from "../provider_info_helpers";
import type { Team } from "../key_team_helpers/key_list";
import type { CredentialItem } from "../networking";
import type { UploadProps } from "antd/es/upload";

// Mock the networking module
vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getGuardrailsList: vi.fn().mockResolvedValue({
      guardrails: [{ guardrail_name: "test-guardrail-1" }, { guardrail_name: "test-guardrail-2" }],
    }),
    tagListCall: vi.fn().mockResolvedValue({}),
    modelAvailableCall: vi.fn().mockResolvedValue({
      data: [{ id: "model-group-1" }, { id: "model-group-2" }],
    }),
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [],
      },
    ]),
  };
});

describe("Add Model Tab", () => {
  it("should render", () => {
    // Create a form instance using renderHook
    const { result } = renderHook(() => Form.useForm());
    const [form] = result.current;

    // Mock functions
    const handleOk = vi.fn();
    const setSelectedProvider = vi.fn();
    const setProviderModelsFn = vi.fn();
    const getPlaceholder = vi.fn((provider: Providers) => `Enter ${provider} model name`);
    const setShowAdvancedSettings = vi.fn();

    // Mock data
    const selectedProvider = Providers.OpenAI;
    const providerModels = ["gpt-4", "gpt-3.5-turbo"];
    const showAdvancedSettings = false;

    const teams: Team[] = [
      {
        team_id: "team-1",
        team_alias: "Test Team",
        models: ["gpt-4"],
        max_budget: 100,
        budget_duration: "monthly",
        tpm_limit: null,
        rpm_limit: null,
        organization_id: "org-1",
        created_at: "2024-01-01T00:00:00Z",
        keys: [],
        members_with_roles: [],
      },
    ];

    const credentials: CredentialItem[] = [
      {
        credential_name: "test-credential",
        credential_values: {},
        credential_info: {
          custom_llm_provider: "openai",
          description: "Test credential",
        },
      },
    ];

    const uploadProps: UploadProps = {
      beforeUpload: () => false,
      showUploadList: false,
    };

    const accessToken = "test-access-token";
    const userRole = "Admin";
    const premiumUser = true;

    const { getByRole } = render(
      <AddModelTab
        form={form}
        handleOk={handleOk}
        selectedProvider={selectedProvider}
        setSelectedProvider={setSelectedProvider}
        providerModels={providerModels}
        setProviderModelsFn={setProviderModelsFn}
        getPlaceholder={getPlaceholder}
        uploadProps={uploadProps}
        showAdvancedSettings={showAdvancedSettings}
        setShowAdvancedSettings={setShowAdvancedSettings}
        teams={teams}
        credentials={credentials}
        accessToken={accessToken}
        userRole={userRole}
        premiumUser={premiumUser}
      />,
    );
    // Check for the heading specifically
    expect(getByRole("heading", { name: "Add Model" })).toBeInTheDocument();
  });
});
