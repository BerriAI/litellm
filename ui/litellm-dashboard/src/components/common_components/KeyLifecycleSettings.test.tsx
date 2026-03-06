import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import KeyLifecycleSettings from "./KeyLifecycleSettings";

vi.mock("antd", () => {
  const Option = ({ children, value }: any) => (
    <option value={value}>{children}</option>
  );
  const Select = ({ children, value, onChange, placeholder }: any) => (
    <select
      data-testid="select"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      data-placeholder={placeholder}
    >
      {children}
    </select>
  );
  Select.Option = Option;
  return {
    Select,
    Tooltip: ({ children, title }: any) => (
      <div data-testid="tooltip" title={title}>
        {children}
      </div>
    ),
    Switch: ({ checked, onChange }: any) => (
      <input
        type="checkbox"
        data-testid="switch"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
    ),
    Divider: () => <hr data-testid="divider" />,
  };
});

vi.mock("@ant-design/icons", () => ({
  InfoCircleOutlined: () => <span data-testid="info-icon">ℹ</span>,
}));

vi.mock("@tremor/react", () => ({
  TextInput: ({ value, onValueChange, onChange, placeholder, name, className }: any) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (onChange) {
        onChange(e);
      }
      if (onValueChange) {
        onValueChange(e.target.value);
      }
    };
    return (
      <input
        data-testid={name === "duration" ? "duration-input" : "custom-interval-input"}
        value={value}
        onChange={handleChange}
        placeholder={placeholder}
        className={className}
      />
    );
  },
}));

describe("KeyLifecycleSettings", () => {
  const mockForm = {
    getFieldValue: vi.fn(),
    setFieldValue: vi.fn(),
    setFieldsValue: vi.fn(),
  };

  const defaultProps = {
    form: mockForm,
    autoRotationEnabled: false,
    onAutoRotationChange: vi.fn(),
    rotationInterval: "",
    onRotationIntervalChange: vi.fn(),
    isCreateMode: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockForm.getFieldValue.mockReturnValue("");
  });

  it("should render without crashing", () => {
    renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

    expect(screen.getByText("Key Expiry Settings")).toBeInTheDocument();
    expect(screen.getByText("Auto-Rotation Settings")).toBeInTheDocument();
  });

  describe("Key Expiry Settings", () => {
    it("should render expiry input field", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      expect(screen.getByText("Expire Key")).toBeInTheDocument();
      expect(screen.getByTestId("duration-input")).toBeInTheDocument();
    });

    it("should show correct placeholder in create mode", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} isCreateMode={true} />);

      const input = screen.getByTestId("duration-input");
      expect(input).toHaveAttribute(
        "placeholder",
        "e.g., 30d or leave empty to never expire"
      );
    });

    it("should show correct placeholder in edit mode", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} isCreateMode={false} />);

      const input = screen.getByTestId("duration-input");
      expect(input).toHaveAttribute("placeholder", "e.g., 30d or -1 to never expire");
    });

    it("should show correct tooltip in create mode", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} isCreateMode={true} />);

      const tooltips = screen.getAllByTestId("tooltip");
      const expiryTooltip = tooltips.find((tooltip) =>
        tooltip.getAttribute("title")?.includes("Leave empty to never expire")
      );
      expect(expiryTooltip).toBeInTheDocument();
      expect(expiryTooltip).toHaveAttribute(
        "title",
        "Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days). Leave empty to never expire."
      );
    });

    it("should show correct tooltip in edit mode", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} isCreateMode={false} />);

      const tooltips = screen.getAllByTestId("tooltip");
      const expiryTooltip = tooltips.find((tooltip) =>
        tooltip.getAttribute("title")?.includes("Use -1 to never expire")
      );
      expect(expiryTooltip).toBeInTheDocument();
      expect(expiryTooltip).toHaveAttribute(
        "title",
        "Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days). Use -1 to never expire."
      );
    });

    it("should initialize with form value if present", () => {
      mockForm.getFieldValue.mockReturnValue("30d");
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      const input = screen.getByTestId("duration-input") as HTMLInputElement;
      expect(input.value).toBe("30d");
    });

    it("should update form using setFieldValue when duration changes", async () => {
      const user = userEvent.setup();
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      const input = screen.getByTestId("duration-input");
      await user.type(input, "60d");

      expect(mockForm.setFieldValue).toHaveBeenCalledWith("duration", "60d");
    });

    it("should update form using setFieldsValue when setFieldValue is not available", async () => {
      const user = userEvent.setup();
      const formWithoutSetFieldValue = {
        getFieldValue: vi.fn().mockReturnValue(""),
        setFieldsValue: vi.fn(),
      };
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} form={formWithoutSetFieldValue} />
      );

      const input = screen.getByTestId("duration-input");
      await user.type(input, "90d");

      expect(formWithoutSetFieldValue.setFieldsValue).toHaveBeenCalledWith({ duration: "90d" });
    });
  });

  describe("Auto-Rotation Settings", () => {
    it("should render auto-rotation switch", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      expect(screen.getByText("Enable Auto-Rotation")).toBeInTheDocument();
      expect(screen.getByTestId("switch")).toBeInTheDocument();
    });

    it("should show switch as unchecked when autoRotationEnabled is false", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} autoRotationEnabled={false} />);

      const switchElement = screen.getByTestId("switch") as HTMLInputElement;
      expect(switchElement.checked).toBe(false);
    });

    it("should show switch as checked when autoRotationEnabled is true", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} />);

      const switchElement = screen.getByTestId("switch") as HTMLInputElement;
      expect(switchElement.checked).toBe(true);
    });

    it("should call onAutoRotationChange when switch is toggled", async () => {
      const user = userEvent.setup();
      const onAutoRotationChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} onAutoRotationChange={onAutoRotationChange} />
      );

      const switchElement = screen.getByTestId("switch");
      await user.click(switchElement);

      expect(onAutoRotationChange).toHaveBeenCalledWith(true);
    });

    it("should not show rotation interval section when auto-rotation is disabled", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} autoRotationEnabled={false} />);

      expect(screen.queryByText("Rotation Interval")).not.toBeInTheDocument();
      expect(screen.queryByTestId("select")).not.toBeInTheDocument();
    });

    it("should show rotation interval section when auto-rotation is enabled", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="30d" />
      );

      expect(screen.getByText("Rotation Interval")).toBeInTheDocument();
      expect(screen.getByTestId("select")).toBeInTheDocument();
    });

    it("should show all predefined interval options", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="30d" />
      );

      expect(screen.getByText("7 days")).toBeInTheDocument();
      expect(screen.getByText("30 days")).toBeInTheDocument();
      expect(screen.getByText("90 days")).toBeInTheDocument();
      expect(screen.getByText("180 days")).toBeInTheDocument();
      expect(screen.getByText("365 days")).toBeInTheDocument();
      expect(screen.getByText("Custom interval")).toBeInTheDocument();
    });

    it("should display current rotation interval in select", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="90d" />
      );

      const select = screen.getByTestId("select") as HTMLSelectElement;
      expect(select.value).toBe("90d");
    });

    it("should call onRotationIntervalChange when predefined interval is selected", async () => {
      const user = userEvent.setup();
      const onRotationIntervalChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="7d"
          onRotationIntervalChange={onRotationIntervalChange}
        />
      );

      const select = screen.getByTestId("select");
      await user.selectOptions(select, "30d");

      expect(onRotationIntervalChange).toHaveBeenCalledWith("30d");
    });

    it("should show custom input when custom option is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="30d" />
      );

      const select = screen.getByTestId("select");
      await user.selectOptions(select, "custom");

      expect(screen.getByTestId("custom-interval-input")).toBeInTheDocument();
      expect(screen.getByText("Supported formats: seconds (s), minutes (m), hours (h), days (d)")).toBeInTheDocument();
    });

    it("should hide custom input when predefined interval is selected after custom", async () => {
      const user = userEvent.setup();
      const onRotationIntervalChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="custom-value"
          onRotationIntervalChange={onRotationIntervalChange}
        />
      );

      const select = screen.getByTestId("select");
      await user.selectOptions(select, "7d");

      expect(screen.queryByTestId("custom-interval-input")).not.toBeInTheDocument();
      expect(onRotationIntervalChange).toHaveBeenCalledWith("7d");
    });

    it("should call onRotationIntervalChange when custom interval is entered", async () => {
      const user = userEvent.setup();
      const onRotationIntervalChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval=""
          onRotationIntervalChange={onRotationIntervalChange}
        />
      );

      const select = screen.getByTestId("select");
      await user.selectOptions(select, "custom");

      const customInput = screen.getByTestId("custom-interval-input");
      await user.type(customInput, "14d");

      expect(onRotationIntervalChange).toHaveBeenCalledWith("14d");
    });

    it("should show info message when auto-rotation is enabled", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} />);

      expect(
        screen.getByText(
          "When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period."
        )
      ).toBeInTheDocument();
    });

    it("should not show info message when auto-rotation is disabled", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} autoRotationEnabled={false} />);

      expect(
        screen.queryByText(
          "When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period."
        )
      ).not.toBeInTheDocument();
    });

    it("should initialize with custom interval input visible when custom interval is provided", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="14d" />
      );

      expect(screen.getByTestId("custom-interval-input")).toBeInTheDocument();
      const customInput = screen.getByTestId("custom-interval-input") as HTMLInputElement;
      expect(customInput.value).toBe("14d");
    });

    it("should show custom option selected when custom interval is provided", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} rotationInterval="14d" />
      );

      const select = screen.getByTestId("select") as HTMLSelectElement;
      expect(select.value).toBe("custom");
    });

    it("should not call onRotationIntervalChange when selecting custom option", async () => {
      const user = userEvent.setup();
      const onRotationIntervalChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="30d"
          onRotationIntervalChange={onRotationIntervalChange}
        />
      );

      const select = screen.getByTestId("select");
      await user.selectOptions(select, "custom");

      expect(onRotationIntervalChange).not.toHaveBeenCalled();
    });
  });
});
