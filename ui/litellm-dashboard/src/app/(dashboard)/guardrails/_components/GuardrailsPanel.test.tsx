import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsPanel from "./GuardrailsPanel";
import { getGuardrailsList, deleteGuardrailCall } from "@/components/networking";

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
  default: ({ guardrailsList, onDeleteClick }: any) => (
    <div>
      <div>Mock Guardrail Table</div>
      {guardrailsList.length > 0 && (
        <button
          data-testid="delete-button"
          onClick={() => onDeleteClick(guardrailsList[0].guardrail_id, guardrailsList[0].guardrail_name)}
        >
          Delete
        </button>
      )}
    </div>
  ),
}));

vi.mock("./guardrail_info", () => ({
  __esModule: true,
  default: () => <div>Mock Guardrail Info View</div>,
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
    render(<GuardrailsPanel {...defaultProps} />);
    expect(screen.getByText("Guardrails")).toBeInTheDocument();
    // Activate the Guardrails tab so its content (including the Add button) is rendered
    fireEvent.click(screen.getByText("Guardrails"));
    expect(screen.getByText("Add New Guardrail")).toBeInTheDocument();
  });

  it("should delete the clicked guardrail after confirming in the modal", async () => {
    render(<GuardrailsPanel {...defaultProps} />);
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
    render(<GuardrailsPanel {...defaultProps} />);

    expect(await screen.findByLabelText("playground draft")).toBeInTheDocument();
    expect(screen.getByText("Mock Team Guardrails Tab")).toBeInTheDocument();
  });

  it("should keep test playground state when switching tabs away and back", async () => {
    render(<GuardrailsPanel {...defaultProps} />);

    fireEvent.click(screen.getByText("Test Playground"));

    const draft = await screen.findByLabelText("playground draft");
    fireEvent.change(draft, { target: { value: "keep me" } });
    expect(draft).toHaveValue("keep me");

    fireEvent.click(screen.getByText("Guardrails"));
    fireEvent.click(screen.getByText("Test Playground"));

    expect(await screen.findByLabelText("playground draft")).toHaveValue("keep me");
  });

  it("should not delete anything when the modal is cancelled", async () => {
    render(<GuardrailsPanel {...defaultProps} />);
    fireEvent.click(screen.getByText("Guardrails"));

    fireEvent.click(await screen.findByTestId("delete-button"));
    const modal = within(await screen.findByRole("dialog"));

    fireEvent.click(modal.getByRole("button", { name: "Cancel" }));

    expect(mockDeleteGuardrailCall).not.toHaveBeenCalled();
  });
});
