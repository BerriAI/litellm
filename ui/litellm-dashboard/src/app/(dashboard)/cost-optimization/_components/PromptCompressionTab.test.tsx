/* @vitest-environment jsdom */
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGetGuardrailsList = vi.fn();
const mockCreateGuardrailCall = vi.fn();
const mockUpdateGuardrailCall = vi.fn();
const mockPush = vi.fn();

vi.mock("@/components/networking", () => ({
  serverRootPath: "/",
  getGuardrailsList: (...args: unknown[]) => mockGetGuardrailsList(...args),
  createGuardrailCall: (...args: unknown[]) => mockCreateGuardrailCall(...args),
  updateGuardrailCall: (...args: unknown[]) => mockUpdateGuardrailCall(...args),
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: mockPush }) }));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

import PromptCompressionTab from "./PromptCompressionTab";

const compressionGuardrail = (overrides: Record<string, unknown> = {}) => ({
  guardrail_id: "fee65a60",
  guardrail_name: "headroom-compression",
  litellm_params: { guardrail: "headroom", api_base: "https://headroom.example.com", default_on: true },
  guardrail_definition_location: "db",
  ...overrides,
});

describe("PromptCompressionTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetGuardrailsList.mockResolvedValue({ guardrails: [compressionGuardrail()] });
  });

  it("switches an always-on endpoint to opt-in without clobbering its other params", async () => {
    mockUpdateGuardrailCall.mockResolvedValue({});
    mockGetGuardrailsList.mockResolvedValueOnce({ guardrails: [compressionGuardrail()] }).mockResolvedValueOnce({
      guardrails: [
        compressionGuardrail({
          litellm_params: { guardrail: "headroom", api_base: "https://headroom.example.com", default_on: false },
        }),
      ],
    });

    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByLabelText("Compression mode for headroom-compression"));
    await user.click(await screen.findByRole("option", { name: "Opt-in" }));

    await waitFor(() =>
      expect(mockUpdateGuardrailCall).toHaveBeenCalledWith("sk-test", "fee65a60", {
        litellm_params: { default_on: false },
      }),
    );
    expect(await screen.findByText(/only runs when a request asks for it/)).toBeInTheDocument();
  });

  it("sends default_on true when switching back to always on", async () => {
    mockUpdateGuardrailCall.mockResolvedValue({});
    mockGetGuardrailsList.mockResolvedValue({
      guardrails: [
        compressionGuardrail({
          litellm_params: { guardrail: "headroom", api_base: "https://headroom.example.com", default_on: false },
        }),
      ],
    });

    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByLabelText("Compression mode for headroom-compression"));
    await user.click(await screen.findByRole("option", { name: "Always on" }));

    await waitFor(() =>
      expect(mockUpdateGuardrailCall).toHaveBeenCalledWith("sk-test", "fee65a60", {
        litellm_params: { default_on: true },
      }),
    );
  });

  it("deep-links Edit settings to the guardrail's settings tab", async () => {
    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    fireEvent.click(await screen.findByRole("button", { name: /Edit settings/ }));

    expect(mockPush).toHaveBeenCalledWith("/ui/guardrails?guardrail=fee65a60&guardrail_tab=settings");
  });

  it("shows a read-only badge instead of edit controls for non-admins", async () => {
    render(<PromptCompressionTab accessToken="sk-test" userRole="Internal User" />);

    expect(await screen.findByText("Always on")).toBeInTheDocument();
    expect(screen.queryByLabelText("Compression mode for headroom-compression")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Edit settings/ })).not.toBeInTheDocument();
  });

  it("keeps a config-file guardrail read-only even for an admin", async () => {
    mockGetGuardrailsList.mockResolvedValue({
      guardrails: [compressionGuardrail({ guardrail_definition_location: "config" })],
    });

    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    expect(await screen.findByText(/Defined in the proxy config file/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Compression mode for headroom-compression")).not.toBeInTheDocument();
  });

  it("creates a guardrail from the empty state and refreshes the list", async () => {
    mockGetGuardrailsList.mockResolvedValueOnce({ guardrails: [] }).mockResolvedValueOnce({
      guardrails: [compressionGuardrail()],
    });
    mockCreateGuardrailCall.mockResolvedValue({});

    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    fireEvent.change(await screen.findByLabelText("Name"), { target: { value: "headroom-compression" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), {
      target: { value: "https://headroom.example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add guardrail" }));

    await waitFor(() =>
      expect(mockCreateGuardrailCall).toHaveBeenCalledWith("sk-test", {
        guardrail_name: "headroom-compression",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://headroom.example.com",
          default_on: true,
        },
      }),
    );
    expect(await screen.findByText("headroom-compression")).toBeInTheDocument();
  });

  it("blocks submission and flags the empty fields instead of creating a blank guardrail", async () => {
    mockGetGuardrailsList.mockResolvedValue({ guardrails: [] });

    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    fireEvent.click(await screen.findByRole("button", { name: "Add guardrail" }));

    expect(await screen.findByText("Name is required")).toBeInTheDocument();
    expect(screen.getByText("API base is required")).toBeInTheDocument();
    expect(mockCreateGuardrailCall).not.toHaveBeenCalled();
  });

  it("hides the add form behind an explicit action once an endpoint exists", async () => {
    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    expect(await screen.findByText("headroom-compression")).toBeInTheDocument();
    expect(screen.queryByLabelText("Headroom API base")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Add another endpoint/ }));

    expect(await screen.findByLabelText("Headroom API base")).toBeInTheDocument();
  });

  it("ignores guardrails from other providers", async () => {
    mockGetGuardrailsList.mockResolvedValue({
      guardrails: [
        compressionGuardrail(),
        {
          guardrail_id: "other",
          guardrail_name: "presidio-pii",
          litellm_params: { guardrail: "presidio", default_on: true },
          guardrail_definition_location: "db",
        },
      ],
    });

    render(<PromptCompressionTab accessToken="sk-test" userRole="Admin" />);

    const list = await screen.findByRole("list");
    expect(within(list).getAllByRole("listitem")).toHaveLength(1);
    expect(screen.queryByText("presidio-pii")).not.toBeInTheDocument();
  });
});
