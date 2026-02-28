import { act, fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import CreateKey from "./create_key_button";

const { formMock, setFieldsValueMock, radioGroupValueRef, formStateRef, mockKeyCreateCall } = vi.hoisted(() => {
  const formStateRef = { current: {} as Record<string, any> };
  const mockKeyCreateCall = vi.fn().mockResolvedValue({
    key: "test-api-key",
    soft_budget: null,
  });
  const formMock = {
    setFieldsValue: vi.fn((values: Record<string, any>) => {
      Object.assign(formStateRef.current, values);
    }),
    setFieldValue: vi.fn((name: string, value: any) => {
      formStateRef.current[name] = value;
    }),
    getFieldValue: vi.fn((name: string) => formStateRef.current[name]),
    resetFields: vi.fn(() => {
      formStateRef.current = {};
    }),
  };
  const radioGroupValueRef = { current: null as string | null };
  return {
    formMock,
    setFieldsValueMock: formMock.setFieldsValue,
    radioGroupValueRef,
    formStateRef,
    mockKeyCreateCall,
  };
});

const defaultAuthorizedState = {
  accessToken: "test-token",
  userId: "test-user-id",
  userRole: "Admin",
  premiumUser: false,
};

let authorizedState = { ...defaultAuthorizedState };

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => authorizedState,
}));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  keyKeys: {
    lists: () => ["keys"],
  },
}));

vi.mock("@ant-design/icons", () => ({
  InfoCircleOutlined: () => null,
}));

vi.mock("react-copy-to-clipboard", () => ({
  CopyToClipboard: ({ children }: { children: any }) => children,
}));

vi.mock("@tremor/react", () => {
  const React = require("react");
  const Stub = ({ children }: { children?: any }) => React.createElement("div", null, children);
  const Button = ({ children, ...props }: { children?: any }) =>
    React.createElement("button", props, children);
  const TextInput = (props: any) => React.createElement("input", props);

  return {
    Accordion: Stub,
    AccordionBody: Stub,
    AccordionHeader: Stub,
    Button,
    Col: Stub,
    Grid: Stub,
    Text: Stub,
    TextInput,
    Title: Stub,
  };
});

vi.mock("antd", () => {
  const React = require("react");

  const getValueFromEvent = (event: any) => {
    if (event?.target) {
      if (event.target.type === "checkbox") {
        return event.target.checked;
      }
      return event.target.value;
    }
    return event;
  };

  const Form = ({ children, onFinish, ...props }: { children?: any; onFinish?: (values: Record<string, any>) => void }) =>
    React.createElement(
      "form",
      {
        ...props,
        onSubmit: (event: Event) => {
          event.preventDefault();
          onFinish?.({ ...formStateRef.current });
        },
      },
      children,
    );

  Form.Item = ({ children, name }: { children?: any; name?: string }) => {
    if (!name || !React.isValidElement(children)) {
      return React.createElement(React.Fragment, null, children);
    }

    return React.cloneElement(children, {
      value: formStateRef.current[name],
      onChange: (event: any) => {
        formStateRef.current[name] = getValueFromEvent(event);
      },
    });
  };

  Form.useForm = () => [formMock];

  const Select = ({ children, onChange, ...props }: { children?: any; onChange?: (value: string) => void }) =>
    React.createElement(
      "select",
      {
        ...props,
        onChange: (event: any) => onChange?.(event.target.value),
      },
      children,
    );

  Select.Option = ({ children, ...props }: { children?: any }) =>
    React.createElement("option", props, children);

  const Input = (props: any) => React.createElement("input", props);
  Input.Password = (props: any) => React.createElement("input", { ...props, type: "password" });
  Input.TextArea = (props: any) => React.createElement("textarea", props);

  const Modal = ({ children, open }: { children?: any; open?: boolean }) =>
    open ? React.createElement("div", null, children) : null;

  const Radio = ({ children, ...props }: { children?: any }) =>
    React.createElement("div", props, children);

  Radio.Group = ({ children, value }: { children?: any; value?: string }) => {
    radioGroupValueRef.current = value ?? null;
    return React.createElement("div", null, children);
  };

  const Switch = (props: any) => React.createElement("input", { ...props, type: "checkbox" });
  const Tag = ({ children }: { children?: any }) => React.createElement("span", null, children);
  const Tooltip = ({ children }: { children?: any }) => React.createElement(React.Fragment, null, children);

  const Button = ({ children, htmlType, ...props }: { children?: any; htmlType?: string }) =>
    React.createElement("button", { ...props, type: htmlType ?? props.type }, children);

  return {
    Button,
    Form,
    Input,
    message: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
    },
    Modal,
    Radio,
    Select,
    Switch,
    Tag,
    Tooltip,
  };
});

vi.mock("../networking", () => ({
  keyCreateCall: mockKeyCreateCall,
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
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
  getAgentsList: vi.fn().mockResolvedValue([]),
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

vi.mock("../agent_management/AgentSelector", () => ({ default: () => null }));
vi.mock("../common_components/budget_duration_dropdown", () => ({ default: () => null }));
vi.mock("../common_components/check_openapi_schema", () => ({ default: () => null }));
vi.mock("../common_components/KeyLifecycleSettings", () => ({ default: () => null }));
vi.mock("../common_components/ModelAliasManager", () => ({ default: () => null }));
vi.mock("../common_components/PassThroughRoutesSelector", () => ({ default: () => null }));
vi.mock("../common_components/PremiumLoggingSettings", () => ({ default: () => null }));
vi.mock("../common_components/RateLimitTypeFormItem", () => ({ default: () => null }));
vi.mock("../common_components/RouterSettingsAccordion", () => ({ default: () => null }));
vi.mock("../common_components/team_dropdown", () => ({ default: () => null }));
vi.mock("../CreateUserButton", () => ({ CreateUserButton: () => null }));
vi.mock("../mcp_server_management/MCPServerSelector", () => ({ default: () => null }));
vi.mock("../mcp_server_management/MCPToolPermissions", () => ({ default: () => null }));
vi.mock("../shared/numerical_input", () => ({ default: () => null }));
vi.mock("../vector_store_management/VectorStoreSelector", () => ({ default: () => null }));
vi.mock("../key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: (model: string) => model,
}));

vi.mock("../common_components/AccessGroupSelector", () => ({
  default: ({ value = [], onChange }: { value?: string[]; onChange?: (v: string[]) => void }) => (
    <input
      data-testid="access-group-selector"
      value={Array.isArray(value) ? value.join(",") : ""}
      onChange={(event) => onChange?.(event.target.value ? event.target.value.split(",").map((v) => v.trim()) : [])}
    />
  ),
}));

describe("CreateKey", () => {
  const defaultProps = {
    team: null,
    teams: [],
    data: [],
    addKey: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    if (typeof window !== "undefined" && window.localStorage && typeof window.localStorage.clear === "function") {
      window.localStorage.clear();
    }
    authorizedState = { ...defaultAuthorizedState };
    radioGroupValueRef.current = null;
    formStateRef.current = {};
    mockKeyCreateCall.mockResolvedValue({
      key: "test-api-key",
      soft_budget: null,
    });
  });

  it("should render the CreateKey component", () => {
    renderWithProviders(<CreateKey {...defaultProps} />);
    expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
  });

  it("should display 'AI APIs' label for the llm_api key type option", async () => {
    renderWithProviders(<CreateKey {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /create new key/i }));
    });

    await waitFor(() => {
      expect(screen.getByText("AI APIs")).toBeInTheDocument();
      expect(screen.queryByText("LLM API")).not.toBeInTheDocument();
    });
  });

  it("should include access_group_ids in keyCreateCall payload when access groups are selected", async () => {
    renderWithProviders(<CreateKey {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /create new key/i }));
    });

    await waitFor(() => {
      expect(screen.getByTestId("access-group-selector")).toBeInTheDocument();
    });

    act(() => {
      fireEvent.change(screen.getByTestId("access-group-selector"), { target: { value: "ag-1,ag-2" } });
      formMock.setFieldValue("key_alias", "Test Key");
    });

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /create key/i }));
    });

    await waitFor(() => {
      expect(mockKeyCreateCall).toHaveBeenCalled();
      const formValues = mockKeyCreateCall.mock.calls[0][2];
      expect(formValues).toHaveProperty("access_group_ids");
      expect(formValues.access_group_ids).toEqual(["ag-1", "ag-2"]);
    });
  });

  it("should prefill models when provided without team_id", async () => {
    renderWithProviders(
      <CreateKey
        {...defaultProps}
        autoOpenCreate={true}
        prefillData={{
          models: ["gpt-4"],
        }}
      />,
    );

    await waitFor(() => {
      expect(setFieldsValueMock).toHaveBeenCalledWith({ models: ["gpt-4"] });
    });
  });

  it("should prefill team_id when it exists in teams", async () => {
    renderWithProviders(
      <CreateKey
        {...defaultProps}
        teams={[{ team_id: "team-1", models: [] } as any]}
        autoOpenCreate={true}
        prefillData={{ team_id: "team-1" }}
      />,
    );

    await waitFor(() => {
      expect(setFieldsValueMock).toHaveBeenCalledWith({ team_id: "team-1" });
    });
  });

  it("should ignore team_id when it does not exist in teams", async () => {
    renderWithProviders(
      <CreateKey
        {...defaultProps}
        teams={[{ team_id: "team-1", models: [] } as any]}
        autoOpenCreate={true}
        prefillData={{ team_id: "team-404", key_alias: "example-key" }}
      />,
    );

    await waitFor(() => {
      expect(setFieldsValueMock).toHaveBeenCalledWith({ key_alias: "example-key" });
    });

    expect(setFieldsValueMock).not.toHaveBeenCalledWith({ team_id: "team-404" });
  });

  it('should fall back to "you" when owned_by is another_user for non-admin', async () => {
    authorizedState = { ...defaultAuthorizedState, userRole: "Internal User" };

    renderWithProviders(
      <CreateKey
        {...defaultProps}
        autoOpenCreate={true}
        prefillData={{ owned_by: "another_user", key_alias: "example-key" }}
      />,
    );

    await waitFor(() => {
      expect(setFieldsValueMock).toHaveBeenCalledWith({ key_alias: "example-key" });
    });

    expect(radioGroupValueRef.current).toBe("you");
  });

  it("should apply owned_by another_user for admin", async () => {
    renderWithProviders(
      <CreateKey
        {...defaultProps}
        autoOpenCreate={true}
        prefillData={{ owned_by: "another_user" }}
      />,
    );

    await waitFor(() => {
      expect(radioGroupValueRef.current).toBe("another_user");
    });
  });

  it("should prefill key_type when provided", async () => {
    renderWithProviders(
      <CreateKey
        {...defaultProps}
        autoOpenCreate={true}
        prefillData={{ key_type: "management" }}
      />,
    );

    await waitFor(() => {
      expect(setFieldsValueMock).toHaveBeenCalledWith({ key_type: "management" });
    });
  });
});
