import { type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsPanel from "./GuardrailsPanel";
import { getGuardrailsList, deleteGuardrailCall } from "@/components/networking";
import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn(),
  deleteGuardrailCall: vi.fn(),
}));

vi.mock("./add_guardrail_form", () => ({
  __esModule: true,
  default: () => <div>Mock Add Guardrail Form</div>,
}));

vi.mock("./guardrail_table", () => ({
  __esModule: true,
  default: ({ guardrailsList, onDeleteClick, onGuardrailClick }: any) => (
    <div>
      <div>Mock Guardrail Table</div>
      {guardrailsList.length > 0 && (
        <>
          <button
            data-testid="delete-button"
            onClick={() => onDeleteClick(guardrailsList[0].guardrail_id, guardrailsList[0].guardrail_name)}
          >
            Delete
          </button>
          <button data-testid="open-button" onClick={() => onGuardrailClick(guardrailsList[0].guardrail_id)}>
            Open
          </button>
        </>
      )}
    </div>
  ),
}));

vi.mock("./guardrail_info", () => ({
  __esModule: true,
  default: ({ guardrailId, onClose }: { guardrailId: string; onClose: () => void }) => (
    <div>
      <div data-testid="guardrail-info-view">Mock Guardrail Info View {guardrailId}</div>
      <button onClick={onClose}>Close Guardrail Info</button>
    </div>
  ),
}));

vi.mock("./GuardrailTestPlayground", async () => {
  const { useState } = await import("react");
  const MockGuardrailTestPlayground = () => {
    const [draft, setDraft] = useState("");
    return (
      <div>
        <div>Mock Guardrail Test Playground</div>
        <input aria-label="playground draft" value={draft} onChange={(e) => setDraft(e.target.value)} />
      </div>
    );
  };
  return { __esModule: true, default: MockGuardrailTestPlayground };
});

vi.mock("./TeamGuardrailsTab", () => ({
  TeamGuardrailsTab: () => <div>Mock Team Guardrails Tab</div>,
}));

vi.mock("@/utils/roles", () => ({
  isAdminRole: vi.fn((role: string) => role === "admin"),
}));

vi.mock("./guardrail_info_helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./guardrail_info_helpers")>()),
  getGuardrailLogoAndName: vi.fn(() => ({
    logo: null,
    displayName: "Test Provider",
  })),
}));

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

describe("GuardrailsPanel", () => {
  const defaultProps = {
    accessToken: "test-token",
    userRole: "admin",
  };

  const mockGetGuardrailsList = vi.mocked(getGuardrailsList);
  const mockDeleteGuardrailCall = vi.mocked(deleteGuardrailCall);

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetGuardrailsList.mockResolvedValue({
      guardrails: [
        {
          guardrail_id: "test-guardrail-1",
          guardrail_name: "Test Guardrail",
          litellm_params: {
            guardrail: "test-provider",
            mode: "async",
            default_on: true,
          },
          guardrail_info: null,
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
          guardrail_definition_location: "database" as any,
        },
      ],
    });
  });

  it("should render the component", async () => {
    renderWithProviders(<GuardrailsPanel {...defaultProps} />);
    expect(screen.getByText("Guardrails")).toBeInTheDocument();
    // Activate the Guardrails tab so its content (including the Add button) is rendered
    fireEvent.click(screen.getByText("Guardrails"));
    expect(screen.getByText("Add New Guardrail")).toBeInTheDocument();
  });

  it("should delete the clicked guardrail after confirming in the modal", async () => {
    renderWithProviders(<GuardrailsPanel {...defaultProps} />);
    fireEvent.click(screen.getByText("Guardrails"));

    fireEvent.click(await screen.findByTestId("delete-button"));

    const modal = within(await screen.findByRole("dialog"));
    expect(modal.getByText("Delete Guardrail")).toBeInTheDocument();
    expect(modal.getByText("test-guardrail-1")).toBeInTheDocument();
    expect(modal.getByText("Test Provider")).toBeInTheDocument();

    fireEvent.click(modal.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(mockDeleteGuardrailCall).toHaveBeenCalledWith("test-token", "test-guardrail-1");
    });
    expect(mockGetGuardrailsList).toHaveBeenCalledTimes(2);
  });

  it("should mount every tab panel up front so panel state survives tab switches", async () => {
    renderWithProviders(<GuardrailsPanel {...defaultProps} />);

    expect(await screen.findByLabelText("playground draft")).toBeInTheDocument();
    expect(screen.getByText("Mock Team Guardrails Tab")).toBeInTheDocument();
  });

  it("should keep test playground state when switching tabs away and back", async () => {
    renderWithProviders(<GuardrailsPanel {...defaultProps} />);

    fireEvent.click(screen.getByText("Test Playground"));

    const draft = await screen.findByLabelText("playground draft");
    fireEvent.change(draft, { target: { value: "keep me" } });
    expect(draft).toHaveValue("keep me");

    fireEvent.click(screen.getByText("Guardrails"));
    fireEvent.click(screen.getByText("Test Playground"));

    expect(await screen.findByLabelText("playground draft")).toHaveValue("keep me");
  });

  it("should not delete anything when the modal is cancelled", async () => {
    renderWithProviders(<GuardrailsPanel {...defaultProps} />);
    fireEvent.click(screen.getByText("Guardrails"));

    fireEvent.click(await screen.findByTestId("delete-button"));
    const modal = within(await screen.findByRole("dialog"));

    fireEvent.click(modal.getByRole("button", { name: "Cancel" }));

    expect(mockDeleteGuardrailCall).not.toHaveBeenCalled();
  });

  describe("guardrail detail deep link (?guardrail=)", () => {
    it("should open the guardrail info view directly from a ?guardrail= deep link", async () => {
      renderWithProviders(<GuardrailsPanel {...defaultProps} />, { searchParams: "?guardrail=test-guardrail-1" });

      expect(await screen.findByTestId("guardrail-info-view")).toHaveTextContent("test-guardrail-1");
      expect(screen.queryByText("Mock Guardrail Table")).not.toBeInTheDocument();
    });

    it("should push ?guardrail= as a new history entry when a guardrail row is clicked", async () => {
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderWithProviders(<GuardrailsPanel {...defaultProps} />, { onUrlUpdate });

      fireEvent.click(await screen.findByTestId("open-button"));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const lastUpdate = onUrlUpdate.mock.calls.at(-1)![0];
      expect(lastUpdate.searchParams.get("guardrail")).toBe("test-guardrail-1");
      expect(lastUpdate.options.history).toBe("push");
      expect(await screen.findByTestId("guardrail-info-view")).toHaveTextContent("test-guardrail-1");
    });

    it("should clear ?guardrail= by replacing history when the info view is closed", async () => {
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderWithProviders(<GuardrailsPanel {...defaultProps} />, {
        searchParams: "?guardrail=test-guardrail-1",
        onUrlUpdate,
      });

      fireEvent.click(await screen.findByRole("button", { name: "Close Guardrail Info" }));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const lastUpdate = onUrlUpdate.mock.calls.at(-1)![0];
      expect(lastUpdate.searchParams.has("guardrail")).toBe(false);
      expect(lastUpdate.options.history).toBe("replace");
      expect(await screen.findByText("Mock Guardrail Table")).toBeInTheDocument();
    });
  });
});
