import { act, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import CreateKey from "./create_key_button";

const { mockKeyCreateCall } = vi.hoisted(() => {
  const fn = vi.fn().mockResolvedValue({
    key: "test-api-key",
    soft_budget: null,
  });
  return { mockKeyCreateCall: fn };
});

vi.mock("../networking", () => ({
  keyCreateCall: mockKeyCreateCall,
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }, { id: "gpt-3.5-turbo" }] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPromptsList: vi.fn().mockResolvedValue({ prompts: [] }),
  proxyBaseUrl: "http://localhost:4000",
  getPossibleUserRoles: vi.fn().mockResolvedValue({
    Admin: { ui_label: "Admin" },
    User: { ui_label: "User" },
  }),
  userFilterUICall: vi.fn().mockResolvedValue([]),
  keyCreateServiceAccountCall: vi.fn().mockResolvedValue({
    key: "test-service-account-key",
    soft_budget: null,
  }),
  fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    clear: vi.fn(),
  },
}));

vi.mock("../common_components/AccessGroupSelector", () => ({
  default: ({ value = [], onChange }: { value?: string[]; onChange?: (v: string[]) => void }) => (
    <input
      data-testid="access-group-selector"
      value={Array.isArray(value) ? value.join(",") : ""}
      onChange={(e) => onChange?.(e.target.value ? e.target.value.split(",").map((s) => s.trim()) : [])}
    />
  ),
}));

describe("CreateKey", () => {
  const defaultProps = {
    team: null,
    data: [],
    teams: [],
    addKey: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockKeyCreateCall.mockResolvedValue({
      key: "test-api-key",
      soft_budget: null,
    });
  });

  it("should render the CreateKey component", () => {
    renderWithProviders(<CreateKey {...defaultProps} />);
    expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
  });

  it("should include access_group_ids in keyCreateCall payload when access groups are selected", async () => {
    renderWithProviders(<CreateKey {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /create new key/i }));
    });

    await waitFor(() => {
      expect(screen.getByLabelText(/key name/i)).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText(/key name/i), { target: { value: "Test Key" } });

    const optionalSettingsAccordion = screen.getByText("Optional Settings");
    act(() => {
      fireEvent.click(optionalSettingsAccordion);
    });

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByTestId("access-group-selector"), { target: { value: "ag-1,ag-2" } });

    const modelsCombobox = screen.getAllByRole("combobox").find((el) => el.closest('[class*="ant-form-item"]')?.textContent?.includes("Models")) ||
      screen.getAllByRole("combobox")[1];
    if (modelsCombobox) {
      act(() => fireEvent.mouseDown(modelsCombobox));
      await waitFor(() => {
        const allTeamModels = [...document.body.querySelectorAll(".ant-select-item")].find(
          (el) => el.textContent?.includes("All Team Models"),
        );
        if (allTeamModels) fireEvent.click(allTeamModels);
      });
    }

    const createButton = screen.getByRole("button", { name: /create key/i });
    act(() => fireEvent.click(createButton));

    await waitFor(
      () => {
        expect(mockKeyCreateCall).toHaveBeenCalled();
        const formValues = mockKeyCreateCall.mock.calls[0][2];
        expect(formValues).toHaveProperty("access_group_ids");
        expect(formValues.access_group_ids).toEqual(["ag-1", "ag-2"]);
      },
      { timeout: 15000 },
    );
  }, { timeout: 30000 });
});
