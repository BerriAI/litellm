import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsPanel from "./GuardrailsPanel";
import {
  deleteGuardrailCall,
  getGuardrailInfo,
  getGuardrailProviderSpecificParams,
  getGuardrailsList,
  getGuardrailUISettings,
} from "@/components/networking";
import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn(),
  deleteGuardrailCall: vi.fn(),
  getGuardrailInfo: vi.fn(),
  getGuardrailUISettings: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("./add_guardrail_form", () => ({ __esModule: true, default: () => null }));
vi.mock("./guardrail_garden", () => ({ __esModule: true, default: () => null }));
vi.mock("./GuardrailTestPlayground", () => ({ __esModule: true, default: () => null }));
vi.mock("./TeamGuardrailsTab", () => ({ TeamGuardrailsTab: () => null }));

const DB_GUARDRAIL = {
  guardrail_id: "db-guardrail-1",
  guardrail_name: "db-word-block",
  litellm_params: { guardrail: "litellm_content_filter", mode: "pre_call", default_on: true },
  guardrail_info: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  guardrail_definition_location: "db",
};

const CONFIG_GUARDRAIL = {
  ...DB_GUARDRAIL,
  guardrail_id: "config-guardrail-1",
  guardrail_name: "cfg-word-block",
  guardrail_definition_location: "config",
};

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

describe("GuardrailsPanel delete from the guardrail detail Settings card", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getGuardrailsList).mockResolvedValue({ guardrails: [DB_GUARDRAIL, CONFIG_GUARDRAIL] });
    vi.mocked(getGuardrailInfo).mockImplementation(async (_token: string, id: string) =>
      id === CONFIG_GUARDRAIL.guardrail_id ? CONFIG_GUARDRAIL : DB_GUARDRAIL,
    );
    vi.mocked(getGuardrailUISettings).mockResolvedValue({
      supported_entities: [],
      supported_actions: [],
      pii_entity_categories: [],
      supported_modes: ["pre_call", "post_call"],
    });
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({});
  });

  const renderDetail = (guardrailId: string) =>
    renderWithProviders(<GuardrailsPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?guardrail=${guardrailId}`,
    });

  it("deletes a DB guardrail only after the admin types its name, then returns to the refreshed table", async () => {
    vi.mocked(deleteGuardrailCall).mockResolvedValue({});
    renderDetail(DB_GUARDRAIL.guardrail_id);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));

    const modal = within(await screen.findByRole("dialog"));
    expect(modal.getByText(/keys, teams, or policies that reference/i)).toBeInTheDocument();
    const confirmButton = modal.getByRole("button", { name: "Delete" });
    expect(confirmButton).toBeDisabled();

    const confirmationInput = modal.getByPlaceholderText(DB_GUARDRAIL.guardrail_name);
    fireEvent.change(confirmationInput, { target: { value: "db-word" } });
    expect(confirmButton).toBeDisabled();
    fireEvent.change(confirmationInput, { target: { value: DB_GUARDRAIL.guardrail_name } });
    expect(confirmButton).toBeEnabled();

    fireEvent.click(confirmButton);

    await waitFor(() => expect(deleteGuardrailCall).toHaveBeenCalledWith("test-token", DB_GUARDRAIL.guardrail_id));
    expect(await screen.findByRole("table")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
    expect(getGuardrailsList).toHaveBeenCalledTimes(2);
  });

  it("keeps the admin on the detail page when the delete request fails", async () => {
    vi.mocked(deleteGuardrailCall).mockRejectedValue(new Error("boom"));
    renderDetail(DB_GUARDRAIL.guardrail_id);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    fireEvent.click(await screen.findByRole("button", { name: "Delete" }));
    const modal = within(await screen.findByRole("dialog"));
    fireEvent.change(modal.getByPlaceholderText(DB_GUARDRAIL.guardrail_name), {
      target: { value: DB_GUARDRAIL.guardrail_name },
    });
    fireEvent.click(modal.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(deleteGuardrailCall).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("does not offer Delete on a config guardrail's detail page", async () => {
    renderDetail(CONFIG_GUARDRAIL.guardrail_id);

    fireEvent.click(await screen.findByRole("tab", { name: "Settings" }));
    expect(await screen.findByRole("img", { name: "Config guardrail details" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
  });
});
