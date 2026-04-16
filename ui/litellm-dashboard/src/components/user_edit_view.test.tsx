import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/test-utils";
import { UserEditView } from "./user_edit_view";

vi.mock("./key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: vi.fn((model: string) => model),
}));

vi.mock("../utils/roles", () => ({
  all_admin_roles: ["Admin", "Admin Viewer", "proxy_admin", "proxy_admin_viewer", "org_admin"],
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const React = await import("react");
  const SelectComponent = ({
    value,
    onChange,
    mode,
    children,
    placeholder,
    disabled,
    style,
    allowClear,
    ...props
  }: any) => {
    const isMultiple = mode === "multiple";
    const selectValue = isMultiple ? (Array.isArray(value) ? value : []) : value || "";
    return React.createElement(
      "select",
      {
        multiple: isMultiple,
        value: selectValue,
        onChange: (e: React.ChangeEvent<HTMLSelectElement>) => {
          const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
          onChange(isMultiple ? selectedValues : selectedValues[0] || undefined);
        },
        disabled,
        placeholder,
        style,
        "aria-label": placeholder || "Select",
        role: "combobox",
        ...props,
      },
      children,
    );
  };
  SelectComponent.displayName = "Select";
  const SelectOption = ({ value: optionValue, children: optionChildren }: any) =>
    React.createElement("option", { value: optionValue }, optionChildren);
  SelectOption.displayName = "SelectOption";
  SelectComponent.Option = SelectOption;
  const Tooltip = ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children);
  Tooltip.displayName = "Tooltip";
  const Checkbox = ({ checked, onChange, children, ...props }: any) =>
    React.createElement(
      "label",
      { style: { display: "flex", alignItems: "center", gap: "8px" } },
      React.createElement("input", {
        type: "checkbox",
        checked: checked,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => onChange({ target: { checked: e.target.checked } }),
        ...props,
      }),
      children,
    );
  Checkbox.displayName = "Checkbox";
  return {
    ...actual,
    Select: SelectComponent,
    Tooltip,
    Checkbox,
  };
});

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  const React = await import("react");
  const SelectItem = ({ value, children, title }: any) => {
    const childText = React.Children.toArray(children)
      .map((child: any) => (typeof child === "string" ? child : child?.props?.children || ""))
      .join(" ");
    return React.createElement("option", { value, title }, childText || title || value);
  };
  SelectItem.displayName = "SelectItem";
  return {
    ...actual,
    SelectItem,
  };
});

describe("UserEditView", () => {
  const MOCK_USER_DATA = {
    user_id: "user-123",
    user_info: {
      user_email: "test@example.com",
      user_alias: "Test User",
      user_role: "proxy_admin",
      models: ["gpt-4", "gpt-3.5-turbo"],
      max_budget: 100.5,
      budget_duration: "30d",
      metadata: {
        key1: "value1",
        key2: "value2",
      },
    },
  };

  const MOCK_POSSIBLE_UI_ROLES = {
    proxy_admin: {
      ui_label: "Proxy Admin",
      description: "Full access to proxy",
    },
    proxy_admin_viewer: {
      ui_label: "Proxy Admin Viewer",
      description: "Read-only access",
    },
    user: {
      ui_label: "User",
      description: "Standard user",
    },
  };

  const defaultProps = {
    userData: MOCK_USER_DATA,
    onCancel: vi.fn(),
    onSubmit: vi.fn(),
    teams: null,
    accessToken: "test-token",
    userID: "current-user-1",
    userRole: "Admin",
    userModels: ["gpt-4", "gpt-3.5-turbo", "claude-3"],
    possibleUIRoles: MOCK_POSSIBLE_UI_ROLES,
    isBulkEdit: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });
  });

  it("should display user ID field when not in bulk edit mode", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("User ID")).toBeInTheDocument();
    });

    const userIdInput = screen.getByLabelText("User ID");
    expect(userIdInput).toBeDisabled();
    expect(userIdInput).toHaveValue("user-123");
  });

  it("should not display user ID field when in bulk edit mode", async () => {
    renderWithProviders(<UserEditView {...defaultProps} isBulkEdit={true} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    expect(screen.queryByLabelText("User ID")).not.toBeInTheDocument();
  });

  it("should display email field when not in bulk edit mode", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("Email")).toBeInTheDocument();
    });

    const emailInput = screen.getByLabelText("Email");
    expect(emailInput).toHaveValue("test@example.com");
  });

  it("should not display email field when in bulk edit mode", async () => {
    renderWithProviders(<UserEditView {...defaultProps} isBulkEdit={true} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    expect(screen.queryByLabelText("Email")).not.toBeInTheDocument();
  });

  it("should display user alias field with initial value", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("User Alias")).toBeInTheDocument();
    });

    const aliasInput = screen.getByLabelText("User Alias");
    expect(aliasInput).toHaveValue("Test User");
  });

  it("should display personal models select with available models", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Personal Models")).toBeInTheDocument();
    });

    const modelsSelect = screen.getByRole("combobox", { name: /select models/i });
    expect(modelsSelect).toBeInTheDocument();
  });

  it("should disable models select when user role is not admin", async () => {
    renderWithProviders(<UserEditView {...defaultProps} userRole="user" />);

    await waitFor(() => {
      const modelsSelect = screen.getByRole("combobox", { name: /select models/i });
      expect(modelsSelect).toBeDisabled();
    });
  });

  it("should enable models select when user role is admin", async () => {
    renderWithProviders(<UserEditView {...defaultProps} userRole="Admin" />);

    await waitFor(() => {
      const modelsSelect = screen.getByRole("combobox", { name: /select models/i });
      expect(modelsSelect).not.toBeDisabled();
    });
  });

  it("should display max budget input field", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Max Budget (USD)")).toBeInTheDocument();
    });
  });

  it("should display unlimited budget checkbox", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("Unlimited Budget")).toBeInTheDocument();
    });
  });

  it("should set unlimited budget checkbox when max_budget is null", async () => {
    const userDataWithNullBudget = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        max_budget: null,
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithNullBudget} />);

    await waitFor(() => {
      const checkbox = screen.getByLabelText("Unlimited Budget");
      expect(checkbox).toBeChecked();
    });
  });

  it("should disable budget input when unlimited budget is checked", async () => {
    const userDataWithNullBudget = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        max_budget: null,
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithNullBudget} />);

    await waitFor(() => {
      const budgetInput = screen.getByRole("spinbutton", { name: /max budget/i });
      expect(budgetInput).toBeDisabled();
    });
  });

  it("should enable budget input when unlimited budget is unchecked", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      const budgetInput = screen.getByRole("spinbutton", { name: /max budget/i });
      expect(budgetInput).not.toBeDisabled();
    });
  });

  it("should clear budget value when unlimited budget is checked", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("Unlimited Budget")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Unlimited Budget");
    await userEvent.click(checkbox);

    await waitFor(() => {
      const budgetInput = screen.getByRole("spinbutton", { name: /max budget/i });
      expect(budgetInput).toHaveValue(null);
    });
  });

  it("should display metadata textarea with formatted JSON", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("Metadata")).toBeInTheDocument();
    });

    const metadataTextarea = screen.getByLabelText("Metadata");
    const expectedJson = JSON.stringify(MOCK_USER_DATA.user_info.metadata, null, 2);
    expect(metadataTextarea).toHaveValue(expectedJson);
  });

  it("should display empty metadata textarea when metadata is undefined", async () => {
    const userDataWithoutMetadata = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        metadata: undefined,
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithoutMetadata} />);

    await waitFor(() => {
      const metadataTextarea = screen.getByLabelText("Metadata");
      expect(metadataTextarea).toHaveValue("");
    });
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const onCancelMock = vi.fn();
    renderWithProviders(<UserEditView {...defaultProps} onCancel={onCancelMock} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await userEvent.click(cancelButton);

    expect(onCancelMock).toHaveBeenCalledTimes(1);
  });

  it("should call onSubmit with form values when form is submitted", async () => {
    const onSubmitMock = vi.fn();
    renderWithProviders(<UserEditView {...defaultProps} onSubmit={onSubmitMock} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });

    const callArgs = onSubmitMock.mock.calls[0][0];
    expect(callArgs.user_id).toBe("user-123");
    expect(callArgs.user_email).toBe("test@example.com");
    expect(callArgs.user_alias).toBe("Test User");
    expect(callArgs.user_role).toBe("proxy_admin");
    expect(callArgs.models).toEqual(["gpt-4", "gpt-3.5-turbo"]);
    expect(callArgs.max_budget).toBe(100.5);
    expect(callArgs.budget_duration).toBe("30d");
    expect(callArgs.metadata).toEqual(MOCK_USER_DATA.user_info.metadata);
  });

  it("should set max_budget to null when unlimited budget is checked on submit", async () => {
    const onSubmitMock = vi.fn();
    const userDataWithNullBudget = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        max_budget: null,
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithNullBudget} onSubmit={onSubmitMock} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });

    const callArgs = onSubmitMock.mock.calls[0][0];
    expect(callArgs.max_budget).toBeNull();
  });

  it("should require budget when unlimited budget is not checked", async () => {
    renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Max Budget (USD)")).toBeInTheDocument();
    });

    const budgetInput = screen.getByRole("spinbutton", { name: /max budget/i });
    await userEvent.clear(budgetInput);

    const checkbox = screen.getByLabelText("Unlimited Budget");
    expect(checkbox).not.toBeChecked();

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText("Please enter a budget or select Unlimited Budget")).toBeInTheDocument();
    });
  });

  it("should allow submission when unlimited budget is checked even if budget is empty", async () => {
    const onSubmitMock = vi.fn();
    renderWithProviders(<UserEditView {...defaultProps} onSubmit={onSubmitMock} />);

    await waitFor(() => {
      expect(screen.getByLabelText("Unlimited Budget")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Unlimited Budget");
    await userEvent.click(checkbox);

    await waitFor(() => {
      expect(checkbox).toBeChecked();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled();
    });
  });

  it("should update form values when userData changes", async () => {
    const { rerender } = renderWithProviders(<UserEditView {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByLabelText("User Alias")).toHaveValue("Test User");
    });

    const updatedUserData = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        user_alias: "Updated Alias",
      },
    };

    rerender(<UserEditView {...defaultProps} userData={updatedUserData} />);

    await waitFor(() => {
      expect(screen.getByLabelText("User Alias")).toHaveValue("Updated Alias");
    });
  });

  it("should handle user data with empty models array", async () => {
    const userDataWithEmptyModels = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        models: [],
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithEmptyModels} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    });

    const submitButton = screen.getByRole("button", { name: /save changes/i });
    await userEvent.click(submitButton);

    await waitFor(() => {
      expect(defaultProps.onSubmit).toHaveBeenCalled();
    });

    const callArgs = defaultProps.onSubmit.mock.calls[0][0];
    expect(callArgs.models).toEqual([]);
  });

  it("should handle user data with undefined max_budget", async () => {
    const userDataWithUndefinedBudget = {
      ...MOCK_USER_DATA,
      user_info: {
        ...MOCK_USER_DATA.user_info,
        max_budget: undefined,
      },
    };

    renderWithProviders(<UserEditView {...defaultProps} userData={userDataWithUndefinedBudget} />);

    await waitFor(() => {
      const checkbox = screen.getByLabelText("Unlimited Budget");
      expect(checkbox).toBeChecked();
    });
  });
});
