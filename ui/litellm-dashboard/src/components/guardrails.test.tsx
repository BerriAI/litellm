import { render, screen } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsPanel from "./guardrails";
import { getGuardrailsList } from "./networking";

vi.mock("./networking", () => ({
  getGuardrailsList: vi.fn(),
  deleteGuardrailCall: vi.fn(),
}));

vi.mock("./guardrails/add_guardrail_form", () => ({
  __esModule: true,
  default: () => <div>Mock Add Guardrail Form</div>,
}));

vi.mock("./guardrails/guardrail_table", () => ({
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

vi.mock("./guardrails/guardrail_info", () => ({
  __esModule: true,
  default: () => <div>Mock Guardrail Info View</div>,
}));

vi.mock("./guardrails/GuardrailTestPlayground", () => ({
  __esModule: true,
  default: () => <div>Mock Guardrail Test Playground</div>,
}));

vi.mock("@/utils/roles", () => ({
  isAdminRole: vi.fn((role: string) => role === "admin"),
}));

vi.mock("./guardrails/guardrail_info_helpers", () => ({
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
    expect(screen.getByText("+ Add New Guardrail")).toBeInTheDocument();
  });
});
