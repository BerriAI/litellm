import * as networking from "@/components/networking";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailInfoView from "./guardrail_info";
import { chooseSelectOption } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getGuardrailInfo: vi.fn(),
  getGuardrailUISettings: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("./content_filter/ContentFilterManager", () => ({
  __esModule: true,
  default: ({ isEditing }: { isEditing: boolean }) => (
    <div data-testid="mock-content-filter-manager">{isEditing && <button>Stray Action</button>}</div>
  ),
  formatContentFilterDataForAPI: () => ({ patterns: [], blocked_words: [], categories: [] }),
}));

const uiSettings = {
  supported_entities: [],
  supported_actions: [],
  pii_entity_categories: [],
  supported_modes: ["pre_call", "post_call"],
};

const bedrockParams = {
  bedrock: {
    guardrailIdentifier: { description: "The guardrail id on Bedrock", required: true, type: null },
    api_key: { description: "API key", required: false, type: null },
    optional_params: {
      description: "Optional parameters",
      required: false,
      type: "nested",
      fields: {
        severity_threshold: { description: "Severity threshold", required: false, type: "number" },
      },
    },
  },
};

const numericProviderParams = {
  bedrock: {
    guardrailIdentifier: { description: "The guardrail id on Bedrock", required: true, type: null },
    max_tokens: { description: "Token ceiling", required: false, type: "number" },
  },
};

const nestedProviderParams = {
  bedrock: {
    guardrailIdentifier: { description: "The guardrail id on Bedrock", required: true, type: null },
    tuning: {
      description: "Nested tuning block",
      required: false,
      type: "nested",
      fields: {
        retries: { description: "How many retries", required: false, type: null },
      },
    },
  },
};

const guardrail = (litellmParams: Record<string, unknown>, guardrailInfo?: Record<string, unknown>) => ({
  guardrail_id: "123",
  guardrail_name: "Test Guardrail",
  litellm_params: { guardrail: "bedrock", mode: "pre_call", default_on: true, ...litellmParams },
  guardrail_info: guardrailInfo,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  guardrail_definition_location: "database",
});

const openEditor = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByText("Settings"));
  await user.click(await screen.findByText("Edit Settings"));
  await screen.findByLabelText("Guardrail Name");
};

const saveChanges = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByText("Save Changes"));
};

const renderView = () => render(<GuardrailInfoView guardrailId="123" onClose={() => {}} accessToken="123" isAdmin />);

const lastPayload = () => vi.mocked(networking.updateGuardrailCall).mock.calls.at(-1)?.[2];

describe("GuardrailInfoView update payload characterization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getGuardrailUISettings).mockResolvedValue(uiSettings);
    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue(bedrockParams);
    vi.mocked(networking.updateGuardrailCall).mockResolvedValue({ status: "success" });
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue(
      guardrail({ guardrailIdentifier: "gr-abc", api_key: "sk-old" }),
    );
  });

  it("sends nothing at all when the user saves without touching a field", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);
    await saveChanges(user);

    await screen.findByText("Edit Settings");
    expect(networking.updateGuardrailCall).not.toHaveBeenCalled();
  });

  it("sends only guardrail_name and drops the empty litellm_params object", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const nameInput = screen.getByLabelText("Guardrail Name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed Guardrail");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ guardrail_name: "Renamed Guardrail" });
  });

  it("never leaks the seeded guardrail, mode or created_at keys into litellm_params", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const identifier = screen.getByLabelText("guardrailIdentifier");
    await user.clear(identifier);
    await user.type(identifier, "gr-new");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { guardrailIdentifier: "gr-new" } });
  });

  it("sends null for a provider param the user cleared", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("api_key"));
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { api_key: null } });
  });

  it("maps the skip system message choice to an explicit boolean", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    await chooseSelectOption(
      user,
      screen.getByLabelText("Skip system messages in guardrail"),
      /exclude from guardrail scan/,
    );
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { skip_system_message_in_guardrail: true } });
  });

  it("parses the guardrail information textarea into an object", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const infoBox = screen.getByLabelText("Guardrail Information");
    await user.clear(infoBox);
    await user.type(infoBox, '{{"team":"platform"}');
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ guardrail_info: { team: "platform" } });
  });

  it("reads a value typed into the optional params section out of the nested optional_params object", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const threshold = screen.getByPlaceholderText("Severity threshold");
    await user.type(threshold, "4");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { severity_threshold: 4 } });
  });

  it("keeps a dotted nested provider field out of the payload entirely", async () => {
    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue(nestedProviderParams);
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    await user.type(screen.getByLabelText("retries"), "7");
    const identifier = screen.getByLabelText("guardrailIdentifier");
    await user.clear(identifier);
    await user.type(identifier, "gr-new");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { guardrailIdentifier: "gr-new" } });
  });

  it("clears a stored nested param to null because no form field ever binds it", async () => {
    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue(nestedProviderParams);
    vi.mocked(networking.getGuardrailInfo).mockResolvedValue(
      guardrail({ guardrailIdentifier: "gr-abc", tuning: { retries: 2 } }),
    );
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const identifier = screen.getByLabelText("guardrailIdentifier");
    await user.clear(identifier);
    await user.type(identifier, "gr-new");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { guardrailIdentifier: "gr-new", tuning: null } });
  });

  it("sends an unnormalised numeric provider field as the raw string the input produced", async () => {
    vi.mocked(networking.getGuardrailProviderSpecificParams).mockResolvedValue(numericProviderParams);
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    await user.type(screen.getByPlaceholderText("Token ceiling"), "5");
    await saveChanges(user);

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ litellm_params: { max_tokens: "5" } });
  });

  it("blocks the save when the required guardrail name is cleared", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    await user.clear(screen.getByLabelText("Guardrail Name"));
    await saveChanges(user);

    await screen.findByText("Please input a guardrail name");
    expect(networking.updateGuardrailCall).not.toHaveBeenCalled();
  });

  it("saves when the user presses Enter in a text field", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const nameInput = screen.getByLabelText("Guardrail Name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed Guardrail{Enter}");

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ guardrail_name: "Renamed Guardrail" });
  });

  it("also saves when any other button inside the form is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const nameInput = screen.getByLabelText("Guardrail Name");
    await user.clear(nameInput);
    await user.type(nameInput, "Renamed Guardrail");
    await user.click(screen.getByText("Stray Action"));

    await waitFor(() => expect(networking.updateGuardrailCall).toHaveBeenCalledTimes(1));
    expect(lastPayload()).toEqual({ guardrail_name: "Renamed Guardrail" });
  });

  it("keeps edits made before Cancel when the editor is reopened", async () => {
    const user = userEvent.setup({ delay: null });
    renderView();
    await openEditor(user);

    const nameInput = screen.getByLabelText("Guardrail Name");
    await user.clear(nameInput);
    await user.type(nameInput, "Draft Name");
    await user.click(screen.getByText("Cancel"));

    await user.click(await screen.findByText("Edit Settings"));
    expect(await screen.findByLabelText("Guardrail Name")).toHaveValue("Draft Name");
  });
});
