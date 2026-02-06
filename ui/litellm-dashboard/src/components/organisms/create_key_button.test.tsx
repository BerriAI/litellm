import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import CreateKey from "./create_key_button";

const { formMock, setFieldsValueMock, radioGroupValueRef } = vi.hoisted(() => {
  const formMock = {
    setFieldsValue: vi.fn(),
    setFieldValue: vi.fn(),
    getFieldValue: vi.fn(),
    resetFields: vi.fn(),
  };
  const radioGroupValueRef = { current: null as string | null };
  return {
    formMock,
    setFieldsValueMock: formMock.setFieldsValue,
    radioGroupValueRef,
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

  const Form = ({ children, ...props }: { children?: any }) =>
    React.createElement("form", props, children);
  Form.Item = ({ children }: { children?: any }) => React.createElement(React.Fragment, null, children);
  Form.useForm = () => [formMock];

  const Select = ({ children, ...props }: { children?: any }) =>
    React.createElement("select", props, children);
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
  const Tooltip = ({ children }: { children?: any }) => React.createElement(React.Fragment, null, children);
  const Button = ({ children, ...props }: { children?: any }) =>
    React.createElement("button", props, children);

  return {
    Button,
    Form,
    Input,
    Modal,
    Radio,
    Select,
    Switch,
    Tooltip,
  };
});

vi.mock("../networking", () => ({
  keyCreateCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }] }),
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
vi.mock("../common_components/team_dropdown", () => ({ default: () => null }));
vi.mock("../create_user_button", () => ({ default: () => null }));
vi.mock("../mcp_server_management/MCPServerSelector", () => ({ default: () => null }));
vi.mock("../mcp_server_management/MCPToolPermissions", () => ({ default: () => null }));
vi.mock("../shared/numerical_input", () => ({ default: () => null }));
vi.mock("../vector_store_management/VectorStoreSelector", () => ({ default: () => null }));
vi.mock("../key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: (model: string) => model,
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
  });

  it("should render the CreateKey component", () => {
    renderWithProviders(<CreateKey {...defaultProps} />);
    expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
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

  it("should fall back to \"you\" when owned_by is another_user for non-admin", async () => {
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
