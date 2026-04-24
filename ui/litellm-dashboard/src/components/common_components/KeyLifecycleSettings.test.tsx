import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import KeyLifecycleSettings from "./KeyLifecycleSettings";

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

  const getDurationInput = () => {
    const inputs = screen.getAllByRole("textbox");
    const duration = inputs.find(
      (el) => (el as HTMLInputElement).name === "duration",
    );
    if (!duration) {
      throw new Error("duration input not found");
    }
    return duration as HTMLInputElement;
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
      expect(getDurationInput()).toBeInTheDocument();
    });

    it("should show correct placeholder in create mode", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} isCreateMode={true} />,
      );

      expect(getDurationInput()).toHaveAttribute(
        "placeholder",
        "e.g., 30d or leave empty to never expire",
      );
    });

    it("should show correct placeholder in edit mode", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} isCreateMode={false} />,
      );

      expect(getDurationInput()).toHaveAttribute("placeholder", "e.g., 30d");
    });

    it("should initialize with form value if present", () => {
      mockForm.getFieldValue.mockReturnValue("30d");
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      expect(getDurationInput().value).toBe("30d");
    });

    it("should update form using setFieldValue when duration changes", async () => {
      const user = userEvent.setup();
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      await user.type(getDurationInput(), "60d");

      expect(mockForm.setFieldValue).toHaveBeenCalledWith("duration", "60d");
    });

    it("should update form using setFieldsValue when setFieldValue is not available", async () => {
      const user = userEvent.setup();
      const formWithoutSetFieldValue = {
        getFieldValue: vi.fn().mockReturnValue(""),
        setFieldsValue: vi.fn(),
      };
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          form={formWithoutSetFieldValue}
        />,
      );

      await user.type(getDurationInput(), "90d");

      expect(formWithoutSetFieldValue.setFieldsValue).toHaveBeenCalledWith({
        duration: "90d",
      });
    });
  });

  describe("Auto-Rotation Settings", () => {
    it("should render auto-rotation switch", () => {
      renderWithProviders(<KeyLifecycleSettings {...defaultProps} />);

      expect(screen.getByText("Enable Auto-Rotation")).toBeInTheDocument();
      expect(screen.getByRole("switch")).toBeInTheDocument();
    });

    it("should show switch as unchecked when autoRotationEnabled is false", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={false}
        />,
      );

      const switchElement = screen.getByRole("switch");
      expect(switchElement).toHaveAttribute("data-state", "unchecked");
    });

    it("should show switch as checked when autoRotationEnabled is true", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} />,
      );

      const switchElement = screen.getByRole("switch");
      expect(switchElement).toHaveAttribute("data-state", "checked");
    });

    it("should call onAutoRotationChange when switch is toggled", async () => {
      const user = userEvent.setup();
      const onAutoRotationChange = vi.fn();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          onAutoRotationChange={onAutoRotationChange}
        />,
      );

      await user.click(screen.getByRole("switch"));

      expect(onAutoRotationChange).toHaveBeenCalledWith(true);
    });

    it("should not show rotation interval section when auto-rotation is disabled", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={false}
        />,
      );

      expect(screen.queryByText("Rotation Interval")).not.toBeInTheDocument();
      expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    });

    it("should show rotation interval section when auto-rotation is enabled", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="30d"
        />,
      );

      expect(screen.getByText("Rotation Interval")).toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("should show all predefined interval options", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="30d"
        />,
      );

      await user.click(screen.getByRole("combobox"));

      expect(
        screen.getByRole("option", { name: "7 days" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "30 days" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "90 days" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "180 days" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "365 days" }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("option", { name: "Custom interval" }),
      ).toBeInTheDocument();
    });

    it("should display current rotation interval in select", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="90d"
        />,
      );

      expect(screen.getByRole("combobox")).toHaveTextContent("90 days");
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
        />,
      );

      await user.click(screen.getByRole("combobox"));
      await user.click(screen.getByRole("option", { name: "30 days" }));

      expect(onRotationIntervalChange).toHaveBeenCalledWith("30d");
    });

    it("should show custom input when custom option is selected", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="30d"
        />,
      );

      await user.click(screen.getByRole("combobox"));
      await user.click(
        screen.getByRole("option", { name: "Custom interval" }),
      );

      expect(
        screen.getByPlaceholderText("e.g., 1s, 5m, 2h, 14d"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(
          "Supported formats: seconds (s), minutes (m), hours (h), days (d)",
        ),
      ).toBeInTheDocument();
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
        />,
      );

      await user.click(screen.getByRole("combobox"));
      await user.click(screen.getByRole("option", { name: "7 days" }));

      expect(
        screen.queryByPlaceholderText("e.g., 1s, 5m, 2h, 14d"),
      ).not.toBeInTheDocument();
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
        />,
      );

      await user.click(screen.getByRole("combobox"));
      await user.click(
        screen.getByRole("option", { name: "Custom interval" }),
      );

      const customInput = screen.getByPlaceholderText("e.g., 1s, 5m, 2h, 14d");
      await user.type(customInput, "14d");

      expect(onRotationIntervalChange).toHaveBeenCalledWith("14d");
    });

    it("should show info message when auto-rotation is enabled", () => {
      renderWithProviders(
        <KeyLifecycleSettings {...defaultProps} autoRotationEnabled={true} />,
      );

      expect(
        screen.getByText(
          "When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period.",
        ),
      ).toBeInTheDocument();
    });

    it("should not show info message when auto-rotation is disabled", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={false}
        />,
      );

      expect(
        screen.queryByText(
          "When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period.",
        ),
      ).not.toBeInTheDocument();
    });

    it("should initialize with custom interval input visible when custom interval is provided", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="14d"
        />,
      );

      const customInput = screen.getByPlaceholderText(
        "e.g., 1s, 5m, 2h, 14d",
      ) as HTMLInputElement;
      expect(customInput).toBeInTheDocument();
      expect(customInput.value).toBe("14d");
    });

    it("should show custom option selected when custom interval is provided", () => {
      renderWithProviders(
        <KeyLifecycleSettings
          {...defaultProps}
          autoRotationEnabled={true}
          rotationInterval="14d"
        />,
      );

      expect(screen.getByRole("combobox")).toHaveTextContent(
        "Custom interval",
      );
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
        />,
      );

      await user.click(screen.getByRole("combobox"));
      await user.click(
        screen.getByRole("option", { name: "Custom interval" }),
      );

      expect(onRotationIntervalChange).not.toHaveBeenCalled();
    });
  });
});
