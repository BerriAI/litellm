import React from "react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

// ---- Hoisted shared mocks (safe to use inside vi.mock factories) ----
const { keyUpdateCallMock, keyDeleteCallMock } = vi.hoisted(() => {
  return {
    keyUpdateCallMock: vi.fn().mockResolvedValue({}),
    keyDeleteCallMock: vi.fn().mockResolvedValue({}),
  };
});

// ---- Module mocks ----

// Networking: wire the hoisted fns so we can assert calls later
vi.mock("../networking", () => {
  return {
    keyUpdateCall: (...args: any[]) => keyUpdateCallMock(...args),
    keyDeleteCall: (...args: any[]) => keyDeleteCallMock(...args),
  };
});

// Notifications
vi.mock("../molecules/notifications_manager", () => {
  const Notifications = {
    success: vi.fn(),
    error: vi.fn(),
    fromBackend: vi.fn(),
  };
  return { default: Notifications };
});

// Roles: ensure 'admin' has write access
vi.mock("../../utils/roles", () => ({
  rolesWithWriteAccess: ["admin"],
}));

// Helpers used in rendering
vi.mock("@/utils/dataUtils", () => ({
  copyToClipboard: async () => true,
  formatNumberWithCommas: (n: any) => String(n),
}));
vi.mock("../key_info_utils", () => ({
  extractLoggingSettings: () => ({}),
  formatMetadataForDisplay: (m: any) => JSON.stringify(m, null, 2),
  stripTagsFromMetadata: (m: any) => m,
}));
vi.mock("../callback_info_helpers", () => ({
  callback_map: {},
  mapInternalToDisplayNames: (x: any) => x,
  mapDisplayToInternalNames: (x: any) => x,
}));
vi.mock("../shared/errorUtils", () => ({
  parseErrorMessage: (e: any) => String(e),
}));

// Tremor components -> async factory, local React import, and named passthroughs
vi.mock("@tremor/react", async () => {
  const React = await import("react");

  const makeNamedPassthrough = (tag: any, name: string) => {
    function Named(props: any) {
      const { children, ...rest } = props;
      return React.createElement(tag, rest, children);
    }
    (Named as any).displayName = name;
    return Named;
  };

  const Card = makeNamedPassthrough("div", "Card");
  const Text = makeNamedPassthrough("span", "Text");
  const Grid = makeNamedPassthrough("div", "Grid");
  const Col = makeNamedPassthrough("div", "Col");
  const TabGroup = makeNamedPassthrough("div", "TabGroup");
  const TabList = makeNamedPassthrough("div", "TabList");
  const TabPanels = makeNamedPassthrough("div", "TabPanels");
  const TabPanel = makeNamedPassthrough("div", "TabPanel");
  const Title = makeNamedPassthrough("h1", "Title");
  const Badge = makeNamedPassthrough("span", "Badge");

  function Button(props: any) {
    const { children, onClick, ...rest } = props;
    return React.createElement("button", { onClick, ...rest }, children);
  }
  (Button as any).displayName = "Button";

  function Tab(props: any) {
    const { children, ...rest } = props;
    return React.createElement("button", { ...rest }, children);
  }
  (Tab as any).displayName = "Tab";

  function TextInput(props: any) {
    return React.createElement("input", { ...props });
  }
  (TextInput as any).displayName = "TextInput";

  function TremorSelect(props: any) {
    return React.createElement("select", { ...props });
  }
  (TremorSelect as any).displayName = "TremorSelect";

  return {
    Card,
    Text,
    Button,
    Grid,
    Col,
    Tab,
    TabList,
    TabGroup,
    TabPanel,
    TabPanels,
    Title,
    Badge,
    TextInput,
    Select: TremorSelect,
  };
});

// antd bits -> async factory & local React
vi.mock("antd", async () => {
  const React = await import("react");

  const Form = { useForm: () => [{}] };

  function Input(props: any) {
    return React.createElement("input", { ...props });
  }
  (Input as any).displayName = "AntdInput";

  function InputNumber(props: any) {
    return React.createElement("input", { ...props });
  }
  (InputNumber as any).displayName = "AntdInputNumber";

  function Select(props: any) {
    return React.createElement("select", { ...props });
  }
  (Select as any).displayName = "AntdSelect";

  function Tooltip({ children }: any) {
    return React.createElement(React.Fragment, null, children);
  }
  (Tooltip as any).displayName = "AntdTooltip";

  function Button(props: any) {
    const { children, onClick, ...rest } = props;
    return React.createElement("button", { onClick, ...rest }, children);
  }
  (Button as any).displayName = "AntdButton";

  return { Form, Input, InputNumber, Select, Tooltip, Button };
});

// Icons -> async factory & local React
vi.mock("@heroicons/react/outline", async () => {
  const React = await import("react");
  function ArrowLeftIcon() {
    return React.createElement("span");
  }
  (ArrowLeftIcon as any).displayName = "ArrowLeftIcon";
  function TrashIcon() {
    return React.createElement("span");
  }
  (TrashIcon as any).displayName = "TrashIcon";
  function RefreshIcon() {
    return React.createElement("span");
  }
  (RefreshIcon as any).displayName = "RefreshIcon";
  return { ArrowLeftIcon, TrashIcon, RefreshIcon };
});

vi.mock("lucide-react", async () => {
  const React = await import("react");
  function CopyIcon() {
    return React.createElement("span");
  }
  (CopyIcon as any).displayName = "CopyIcon";
  function CheckIcon() {
    return React.createElement("span");
  }
  (CheckIcon as any).displayName = "CheckIcon";
  return { CopyIcon, CheckIcon };
});

// Heavy children -> async factories & local React
vi.mock("../organisms/regenerate_key_modal", async () => {
  const React = await import("react");
  function RegenerateKeyModal() {
    return null;
  }
  (RegenerateKeyModal as any).displayName = "RegenerateKeyModal";
  return { RegenerateKeyModal };
});
vi.mock("../object_permissions_view", async () => {
  const React = await import("react");
  function ObjectPermissionsView() {
    return null;
  }
  (ObjectPermissionsView as any).displayName = "ObjectPermissionsView";
  return { __esModule: true, default: ObjectPermissionsView };
});
vi.mock("../logging_settings_view", async () => {
  const React = await import("react");
  function LoggingSettingsView() {
    return null;
  }
  (LoggingSettingsView as any).displayName = "LoggingSettingsView";
  return { __esModule: true, default: LoggingSettingsView };
});
vi.mock("../common_components/AutoRotationView", async () => {
  const React = await import("react");
  function AutoRotationView() {
    return null;
  }
  (AutoRotationView as any).displayName = "AutoRotationView";
  return { __esModule: true, default: AutoRotationView };
});

// KeyEditView mock: triggers onSubmit with our injected form values
vi.mock("./key_edit_view", async () => {
  const React = await import("react");
  function KeyEditView(props: any) {
    return React.createElement(
      "div",
      null,
      React.createElement(
        "button",
        {
          onClick: () => props.onSubmit((globalThis as any).__TEST_FORM_VALUES ?? {}),
        },
        "Mock Submit",
      ),
    );
  }
  (KeyEditView as any).displayName = "KeyEditViewMock";
  return { KeyEditView };
});

// ---- SUT import AFTER mocks ----
import KeyInfoView from "./key_info_view";

// ---- Test data helpers ----
const baseKeyData = {
  token_id: "tok_123",
  token: "tok_123",
  key_alias: "My API Key",
  key_name: "sk-xxxx",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  spend: 0,
  max_budget: null,
  tpm_limit: null,
  rpm_limit: null,
  models: [] as string[],
  metadata: {} as Record<string, any>,
  object_permission: {} as Record<string, any>,
  auto_rotate: false,
  rotation_interval: null as any,
  last_rotation_at: null as any,
  key_rotation_at: null as any,
  next_rotation_at: null as any,
};

const renderView = (premiumUser: boolean) =>
  render(
    <KeyInfoView
      keyId="tok_123"
      onClose={() => {}}
      keyData={baseKeyData as any}
      onKeyDataUpdate={() => {}}
      accessToken="access_abc"
      userID="user_1"
      userRole="admin"
      teams={[]}
      premiumUser={premiumUser}
      setAccessToken={() => {}}
    />,
  );

beforeEach(() => {
  vi.clearAllMocks();
  (globalThis as any).__TEST_FORM_VALUES = undefined;
});

// ---- Tests ----
describe("KeyInfoView handleKeyUpdate premium guard", () => {
  it("removes guardrails & prompts for non-premium users and prevents metadata.guardrails", async () => {
    renderView(false); // premiumUser = false

    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      guardrails: ["gr-1", "gr-2"],
      prompts: ["fast", "safe"],
      metadata: {}, // object form (not JSON string)
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [sentAccessToken, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentAccessToken).toBe("access_abc");

    expect("guardrails" in sentPayload).toBe(false);
    expect("prompts" in sentPayload).toBe(false);
    expect(sentPayload.metadata?.guardrails).toBeUndefined();
    expect(sentPayload.key).toBe("tok_123");
  });

  it("preserves guardrails & prompts for premium users and includes metadata.guardrails", async () => {
    renderView(true); // premiumUser = true

    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      guardrails: ["gr-1"],
      prompts: ["fast"],
      metadata: {},
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];

    expect(sentPayload.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.prompts).toEqual(["fast"]);
    expect(sentPayload.metadata?.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.key).toBe("tok_123");
  });
});

describe("KeyInfoView handleKeyUpdate empty strings", () => {
  ["tpm_limit", "rpm_limit", "max_parallel_requests", "max_budget"].forEach((limit) => {
    it(`maps empty strings to null for ${limit}`, async () => {
      renderView(true); // premiumUser = true

      fireEvent.click(screen.getByText("Edit Settings"));
      (globalThis as any).__TEST_FORM_VALUES = {
        token: "tok_123",
        [limit]: "",
      };

      fireEvent.click(screen.getByText("Mock Submit"));

      await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

      const [sentAccessToken, sentPayload] = keyUpdateCallMock.mock.calls[0];
      expect(sentAccessToken).toBe("access_abc");
      expect(sentPayload[limit]).toBeNull();
    });
  });
});
