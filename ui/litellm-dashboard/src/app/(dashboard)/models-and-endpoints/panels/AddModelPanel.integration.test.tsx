import { screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import AddModelPanel from "./AddModelPanel";

const ptuEnabled = vi.fn<() => boolean>();

vi.mock("@/components/add_model/handle_add_model_submit", () => ({
  handleAddModelSubmit: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("@/lib/toast", () => ({
  toast: { fromError: vi.fn(), success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() },
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "test-token",
    accessToken: "test-access-token",
    userId: "user-1",
    userEmail: "test@example.com",
    userRole: "proxy_admin",
    premiumUser: true,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: {}, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => ({ data: { credentials: [] }, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [], isLoading: false, error: null }),
  useInfiniteTeams: () => ({
    data: { pages: [{ teams: [], total: 0, page: 1, page_size: 20, total_pages: 1 }] },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled", () => ({
  usePtuCostAttributionEnabled: () => ptuEnabled(),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrails", () => ({
  useGuardrails: () => ({
    data: {
      guardrails: [{ guardrail_name: "test-guardrail" }],
      globalGuardrailNames: new Set<string>(),
      optionalGuardrailNames: new Set<string>(["test-guardrail"]),
    },
    isLoading: false,
    error: null,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/tags/useTags", () => ({
  useTags: () => ({ data: {}, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/providers/useProviderFields", () => ({
  useProviderFields: () => ({
    data: [
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-3.5-turbo",
        credential_fields: [],
      },
    ],
    isLoading: false,
    error: null,
  }),
}));

vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  default: () => <div data-testid="vector-store-selector" />,
}));

vi.mock("@/components/networking", async () => {
  const actual = await vi.importActual<Record<string, unknown>>("@/components/networking");
  return {
    ...actual,
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
    modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
    getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
    tagListCall: vi.fn().mockResolvedValue({}),
  };
});

import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";

type Payload = Record<string, unknown>;

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const fillRequiredFields = async (user: ReturnType<typeof setup>) => {
  await screen.findByRole("heading", { name: "Add Model" });
  await user.click(screen.getByText("Select a provider"));
  await user.click((await screen.findAllByRole("option"))[0]);
  await user.type(screen.getByPlaceholderText("gpt-3.5-turbo"), "gpt-4o");
};

const submitAndCapturePayload = async (user: ReturnType<typeof setup>): Promise<Payload> => {
  await user.click(screen.getByTestId("add-model-btn"));
  await waitFor(() => expect(handleAddModelSubmit).toHaveBeenCalled());
  return vi.mocked(handleAddModelSubmit).mock.calls[0][0] as Payload;
};

const REQUIRED_FIELD_KEYS = {
  custom_llm_provider: "OpenAI",
  litellm_credential_name: null,
  mode: undefined,
  model: "gpt-4o",
  model_access_group: undefined,
  model_mappings: [{ litellm_model: "gpt-4o", public_name: "gpt-4o" }],
};

const ADVANCED_SETTINGS_DEFAULT_KEYS = {
  cache_control: undefined,
  custom_pricing: undefined,
  guardrails: undefined,
  litellm_extra_params: undefined,
  model_info_params: undefined,
  tags: undefined,
  use_in_pass_through: undefined,
  vector_store_ids: undefined,
};

const EXPANDED_PAYLOAD = { ...REQUIRED_FIELD_KEYS, ...ADVANCED_SETTINGS_DEFAULT_KEYS };

const PTU_ENABLED_PAYLOAD = {
  ...EXPANDED_PAYLOAD,
  ptu_count: undefined,
  cost_per_ptu_per_hour: undefined,
  ptu_effective_from: undefined,
  ptu_effective_to: undefined,
};

const TYPED_VALUES_PAYLOAD = {
  ...EXPANDED_PAYLOAD,
  use_in_pass_through: true,
  litellm_extra_params: '{"rpm": 7}',
  model_info_params: '{"mode": "chat"}',
};

const CUSTOM_PRICING_PAYLOAD = {
  ...EXPANDED_PAYLOAD,
  custom_pricing: true,
  pricing_model: undefined,
  input_cost_per_token: "42",
  output_cost_per_token: undefined,
  cache_read_input_token_cost: undefined,
  cache_creation_input_token_cost: undefined,
};

const PER_SECOND_PRICING_PAYLOAD = {
  ...EXPANDED_PAYLOAD,
  custom_pricing: true,
  pricing_model: "per_second",
  input_cost_per_second: undefined,
};

describe("AddModelPanel submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ptuEnabled.mockReturnValue(false);
  });

  it("sends only the always-mounted fields while Advanced Settings stays collapsed", async () => {
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);

    expect(await submitAndCapturePayload(user)).toStrictEqual(REQUIRED_FIELD_KEYS);
  });

  it("adds every Advanced Settings field once the section is expanded", async () => {
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("LiteLLM Params");

    expect(await submitAndCapturePayload(user)).toStrictEqual(EXPANDED_PAYLOAD);
  });

  it("adds the PTU fields only when cost attribution is enabled, which no click can reach", async () => {
    ptuEnabled.mockReturnValue(true);
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("PTU Count");

    expect(await submitAndCapturePayload(user)).toStrictEqual(PTU_ENABLED_PAYLOAD);
  });

  it("carries values typed into Advanced Settings through to the payload", async () => {
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("LiteLLM Params");

    await user.click(screen.getByLabelText(/Use in pass through routes/i));
    await user.type(screen.getByPlaceholderText(/"rpm": 100/s), '{{"rpm": 7}');
    await user.type(screen.getByPlaceholderText(/"mode": "chat"/s), '{{"mode": "chat"}');

    expect(await submitAndCapturePayload(user)).toStrictEqual(TYPED_VALUES_PAYLOAD);
  });

  it("keeps the pricing fields out of the payload until Custom Pricing is switched on", async () => {
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("LiteLLM Params");

    await user.click(screen.getByLabelText(/Custom Pricing/i));
    await screen.findByText("Pricing Model");
    await user.type(screen.getByLabelText(/Input Cost \(per 1M tokens\)/i), "42");

    expect(await submitAndCapturePayload(user)).toStrictEqual(CUSTOM_PRICING_PAYLOAD);
  });

  it("swaps the per-token cost fields for the per-second one when the pricing model changes", async () => {
    const user = setup();
    renderWithProviders(<AddModelPanel />);
    await fillRequiredFields(user);
    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("LiteLLM Params");

    await user.click(screen.getByLabelText(/Custom Pricing/i));
    await user.click(await screen.findByText("Per Million Tokens"));
    await user.click(await screen.findByTitle("Per Second"));
    await screen.findByText("Cost Per Second");

    expect(await submitAndCapturePayload(user)).toStrictEqual(PER_SECOND_PRICING_PAYLOAD);
  });
});
