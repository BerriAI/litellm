import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import CreateKey from "./create_key_button";

const mockKeyCreateCall = vi.fn().mockResolvedValue({
  key: "test-api-key",
  soft_budget: null,
});

vi.mock("./networking", () => ({
  keyCreateCall: mockKeyCreateCall,
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
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

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    clear: vi.fn(),
  },
}));

describe("CreateKey", () => {
  const defaultProps = {
    userID: "test-user-id",
    team: null,
    userRole: "Admin",
    accessToken: "test-token",
    data: [],
    teams: [],
    addKey: vi.fn(),
    premiumUser: false,
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
    render(<CreateKey {...defaultProps} />);
    expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
  });

  it("should keep duration as null when nothing is inputted", async () => {
    const addKey = vi.fn();
    render(<CreateKey {...defaultProps} addKey={addKey} />);

    const createButton = screen.getByRole("button", { name: /create new key/i });
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Key Ownership")).toBeInTheDocument();
    });

    const keyAliasInput = screen.getByPlaceholderText("");
    act(() => {
      fireEvent.change(keyAliasInput, { target: { value: "test-key" } });
    });

    const modelsSelect = screen.getByPlaceholderText("Select models");
    act(() => {
      fireEvent.mouseDown(modelsSelect);
    });

    await waitFor(() => {
      const allTeamModelsOption = screen.getByText("All Team Models");
      act(() => {
        fireEvent.click(allTeamModelsOption);
      });
    });

    const submitButton = screen.getByRole("button", { name: /create key/i });

    let formValues: Record<string, any> = {};
    mockKeyCreateCall.mockImplementation(async (_token: string, _userId: string, values: Record<string, any>) => {
      formValues = values;
      return { key: "test-api-key", soft_budget: null };
    });

    act(() => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(addKey).toHaveBeenCalled();
    });

    expect(formValues.duration).toBeNull();
  }, 10000); // 10 second timeout for complex test

  it("should set duration correctly when a value is provided", async () => {
    const addKey = vi.fn();
    render(<CreateKey {...defaultProps} addKey={addKey} />);

    const createButton = screen.getByRole("button", { name: /create new key/i });
    act(() => {
      fireEvent.click(createButton);
    });

    await waitFor(() => {
      expect(screen.getByText("Key Ownership")).toBeInTheDocument();
    });

    const keyAliasInput = screen.getByPlaceholderText("");
    act(() => {
      fireEvent.change(keyAliasInput, { target: { value: "test-key" } });
    });

    const modelsSelect = screen.getByPlaceholderText("Select models");
    act(() => {
      fireEvent.mouseDown(modelsSelect);
    });

    await waitFor(() => {
      const allTeamModelsOption = screen.getByText("All Team Models");
      act(() => {
        fireEvent.click(allTeamModelsOption);
      });
    });

    const optionalSettingsAccordion = screen.getByText("Optional Settings");
    act(() => {
      fireEvent.click(optionalSettingsAccordion);
    });

    await waitFor(() => {
      const keyLifecycleAccordion = screen.getByText("Key Lifecycle");
      act(() => {
        fireEvent.click(keyLifecycleAccordion);
      });
    });

    await waitFor(() => {
      const durationInput = screen.getByPlaceholderText("e.g., 30d");
      act(() => {
        fireEvent.change(durationInput, { target: { value: "30d" } });
      });
    });

    const submitButton = screen.getByRole("button", { name: /create key/i });

    let formValues: Record<string, any> = {};
    mockKeyCreateCall.mockImplementation(async (_token: string, _userId: string, values: Record<string, any>) => {
      formValues = values;
      return { key: "test-api-key", soft_budget: null };
    });

    act(() => {
      fireEvent.click(submitButton);
    });

    await waitFor(() => {
      expect(addKey).toHaveBeenCalled();
    });

    expect(formValues.duration).toBe("30d");
  });
}, 10000); // 10 second timeout for complex test
