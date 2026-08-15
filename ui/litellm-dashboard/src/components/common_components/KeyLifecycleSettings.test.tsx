import React, { useState } from "react";
// eslint-disable-next-line no-restricted-imports -- exercising KeyLifecycleSettings requires hosting it in a real antd Form (the component it's built on)
import { Form } from "antd";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import KeyLifecycleSettings from "./KeyLifecycleSettings";

const CREATE_PLACEHOLDER = "e.g., 30d or leave empty to never expire";
const EDIT_PLACEHOLDER = "e.g., 30d";

interface HarnessProps {
  isCreateMode?: boolean;
  onFinish?: (values: Record<string, unknown>) => void;
}

const Harness: React.FC<HarnessProps> = ({ isCreateMode = true, onFinish = () => {} }) => {
  const [form] = Form.useForm();
  const [autoRotationEnabled, setAutoRotationEnabled] = useState(false);
  const [rotationInterval, setRotationInterval] = useState("");
  const [neverExpire, setNeverExpire] = useState(false);

  return (
    <Form form={form} onFinish={onFinish}>
      <KeyLifecycleSettings
        form={form}
        autoRotationEnabled={autoRotationEnabled}
        onAutoRotationChange={setAutoRotationEnabled}
        rotationInterval={rotationInterval}
        onRotationIntervalChange={setRotationInterval}
        isCreateMode={isCreateMode}
        neverExpire={neverExpire}
        onNeverExpireChange={setNeverExpire}
      />
      <button type="submit">submit</button>
      <button type="button" onClick={() => form.resetFields()}>
        reset
      </button>
      <span data-testid="rotation-interval-value">{rotationInterval}</span>
    </Form>
  );
};

const getDurationInput = (isCreateMode = true) =>
  screen.getByPlaceholderText(isCreateMode ? CREATE_PLACEHOLDER : EDIT_PLACEHOLDER) as HTMLInputElement;

describe("KeyLifecycleSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the expiry and auto-rotation sections", () => {
    renderWithProviders(<Harness />);
    expect(screen.getByText("Key Expiry Settings")).toBeInTheDocument();
    expect(screen.getByText("Auto-Rotation Settings")).toBeInTheDocument();
    expect(getDurationInput()).toBeInTheDocument();
  });

  it("uses the create-mode placeholder in create mode", () => {
    renderWithProviders(<Harness isCreateMode={true} />);
    expect(screen.getByPlaceholderText(CREATE_PLACEHOLDER)).toBeInTheDocument();
  });

  it("uses the edit-mode placeholder in edit mode", () => {
    renderWithProviders(<Harness isCreateMode={false} />);
    expect(screen.getByPlaceholderText(EDIT_PLACEHOLDER)).toBeInTheDocument();
  });

  describe("duration is a single source of truth (regression for pre-filled value dropped on submit)", () => {
    it("submits the duration the user typed", async () => {
      const user = userEvent.setup();
      const onFinish = vi.fn();
      renderWithProviders(<Harness onFinish={onFinish} />);

      await user.type(getDurationInput(), "1d");
      await user.click(screen.getByRole("button", { name: "submit" }));

      await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1));
      expect(onFinish.mock.calls[0][0]).toMatchObject({ duration: "1d" });
    });

    it("clears the displayed value when the form is reset, so no stale value lingers", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await user.type(getDurationInput(), "1d");
      expect(getDurationInput().value).toBe("1d");

      await user.click(screen.getByRole("button", { name: "reset" }));

      await waitFor(() => expect(getDurationInput().value).toBe(""));
    });

    it("never submits a value that differs from what is displayed after a reset", async () => {
      const user = userEvent.setup();
      const onFinish = vi.fn();
      renderWithProviders(<Harness onFinish={onFinish} />);

      // First create: type "1d" and submit -> "1d" is sent.
      await user.type(getDurationInput(), "1d");
      await user.click(screen.getByRole("button", { name: "submit" }));
      await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1));
      expect(onFinish.mock.calls[0][0]).toMatchObject({ duration: "1d" });

      // Second create: form resets, so the field must show empty AND submit empty.
      // The old bug showed a stale "1d" while submitting null/empty.
      await user.click(screen.getByRole("button", { name: "reset" }));
      await waitFor(() => expect(getDurationInput().value).toBe(""));

      await user.click(screen.getByRole("button", { name: "submit" }));
      await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(2));
      expect(onFinish.mock.calls[1][0].duration).not.toBe("1d");
      expect(getDurationInput().value).toBe(onFinish.mock.calls[1][0].duration ?? "");
    });
  });

  describe("Never Expire", () => {
    it("clears and disables the duration input, then submits an empty duration", async () => {
      const user = userEvent.setup();
      const onFinish = vi.fn();
      renderWithProviders(<Harness isCreateMode={false} onFinish={onFinish} />);

      await user.type(getDurationInput(false), "30d");
      expect(getDurationInput(false).value).toBe("30d");

      await user.click(screen.getByRole("checkbox", { name: /never expire/i }));

      await waitFor(() => expect(getDurationInput(false).value).toBe(""));
      expect(getDurationInput(false)).toBeDisabled();

      await user.click(screen.getByRole("button", { name: "submit" }));
      await waitFor(() => expect(onFinish).toHaveBeenCalledTimes(1));
      expect(onFinish.mock.calls[0][0]).toMatchObject({ duration: "" });
    });
  });

  describe("Auto-Rotation", () => {
    it("reveals the rotation interval controls when enabled", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      expect(screen.queryByText("Rotation Interval")).not.toBeInTheDocument();
      await user.click(screen.getByRole("switch"));

      await waitFor(() => expect(screen.getByText("Rotation Interval")).toBeInTheDocument());
    });

    it("propagates a selected predefined interval", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(screen.getByText("Rotation Interval")).toBeInTheDocument());

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("90 days"));

      await waitFor(() => expect(document.querySelector(".ant-select-selection-item")?.textContent).toBe("90 days"));
      expect(screen.getByTestId("rotation-interval-value")).toHaveTextContent("90d");
    });

    it("shows the custom interval input when Custom interval is selected, without propagating yet", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(screen.getByText("Rotation Interval")).toBeInTheDocument());

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("Custom interval"));

      expect(await screen.findByPlaceholderText("e.g., 1s, 5m, 2h, 14d")).toBeInTheDocument();
      expect(screen.getByText("Supported formats: seconds (s), minutes (m), hours (h), days (d)")).toBeInTheDocument();
      expect(screen.getByTestId("rotation-interval-value")).toHaveTextContent("");
    });

    it("propagates a typed custom interval to the parent", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(screen.getByText("Rotation Interval")).toBeInTheDocument());

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("Custom interval"));

      const customInput = await screen.findByPlaceholderText("e.g., 1s, 5m, 2h, 14d");
      await user.type(customInput, "14d");

      await waitFor(() => expect(screen.getByTestId("rotation-interval-value")).toHaveTextContent("14d"));
      expect((customInput as HTMLInputElement).value).toBe("14d");
    });

    it("hides the custom input and propagates the value when switching back to a predefined interval", async () => {
      const user = userEvent.setup();
      renderWithProviders(<Harness />);

      await user.click(screen.getByRole("switch"));
      await waitFor(() => expect(screen.getByText("Rotation Interval")).toBeInTheDocument());

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("Custom interval"));
      const customInput = await screen.findByPlaceholderText("e.g., 1s, 5m, 2h, 14d");
      await user.type(customInput, "14d");
      await waitFor(() => expect(screen.getByTestId("rotation-interval-value")).toHaveTextContent("14d"));

      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("7 days"));

      await waitFor(() => expect(screen.getByTestId("rotation-interval-value")).toHaveTextContent("7d"));
      expect(screen.queryByPlaceholderText("e.g., 1s, 5m, 2h, 14d")).not.toBeInTheDocument();
    });
  });
});
