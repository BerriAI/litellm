import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { UnifiedSelector } from "./UnifiedSelector";
import { EndpointId, ENDPOINT_CONFIGS } from "../endpoint_config";

describe("UnifiedSelector", () => {
  it("should render", () => {
    const onChange = vi.fn();
    const options = [
      { value: "option1", label: "Option 1" },
      { value: "option2", label: "Option 2" },
    ];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    render(<UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />);

    const select = screen.getByRole("combobox");
    expect(select).toBeInTheDocument();
  });

  it("should display placeholder when not loading", () => {
    const onChange = vi.fn();
    const options = [{ value: "option1", label: "Option 1" }];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    const { container } = render(
      <UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />,
    );

    const placeholder = container.querySelector(".ant-select-selection-placeholder");
    expect(placeholder).toHaveTextContent(config.selectorPlaceholder);
  });

  it("should display loading placeholder when loading", () => {
    const onChange = vi.fn();
    const options = [{ value: "option1", label: "Option 1" }];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    const { container } = render(
      <UnifiedSelector value="" options={options} loading={true} config={config} onChange={onChange} />,
    );

    const placeholder = container.querySelector(".ant-select-selection-placeholder");
    expect(placeholder).toHaveTextContent(`Loading ${config.selectorLabel.toLowerCase()}s...`);
  });

  it("should call onChange when option is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const options = [
      { value: "option1", label: "Option 1" },
      { value: "option2", label: "Option 2" },
    ];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    render(<UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    await waitFor(() => {
      const option = screen.getByText("Option 1");
      expect(option).toBeInTheDocument();
    });

    const option = screen.getByText("Option 1");
    await user.click(option);

    await waitFor(() => {
      expect(onChange).toHaveBeenCalled();
    });
    const callArgs = onChange.mock.calls[0];
    expect(callArgs[0]).toBe("option1");
  });

  it("should display selected value", () => {
    const onChange = vi.fn();
    const options = [
      { value: "option1", label: "Option 1" },
      { value: "option2", label: "Option 2" },
    ];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    const { container } = render(
      <UnifiedSelector value="option1" options={options} loading={false} config={config} onChange={onChange} />,
    );

    const selectedValue = container.querySelector(".ant-select-selection-item");
    expect(selectedValue).toHaveTextContent("Option 1");
  });

  it("should filter options by search input", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const options = [
      { value: "option1", label: "Option One" },
      { value: "option2", label: "Option Two" },
      { value: "option3", label: "Different" },
    ];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    render(<UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);
    await user.type(select, "One");

    await waitFor(() => {
      expect(screen.getByText("Option One")).toBeInTheDocument();
      expect(screen.queryByText("Option Two")).not.toBeInTheDocument();
      expect(screen.queryByText("Different")).not.toBeInTheDocument();
    });
  });

  it("should show loading spinner in notFoundContent when loading", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const options: { value: string; label: string }[] = [];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    render(<UnifiedSelector value="" options={options} loading={true} config={config} onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    await waitFor(() => {
      const spin = document.querySelector(".ant-spin");
      expect(spin).toBeInTheDocument();
    });
  });

  it("should show no options message when not loading and no options", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const options: { value: string; label: string }[] = [];
    const config = ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS];

    render(<UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />);

    const select = screen.getByRole("combobox");
    await user.click(select);

    await waitFor(() => {
      expect(screen.getByText(`No ${config.selectorLabel.toLowerCase()}s available`)).toBeInTheDocument();
    });
  });

  it("should work with agent endpoint config", () => {
    const onChange = vi.fn();
    const options = [{ value: "agent1", label: "Agent One" }];
    const config = ENDPOINT_CONFIGS[EndpointId.A2A_AGENTS];

    const { container } = render(
      <UnifiedSelector value="" options={options} loading={false} config={config} onChange={onChange} />,
    );

    const placeholder = container.querySelector(".ant-select-selection-placeholder");
    expect(placeholder).toHaveTextContent(config.selectorPlaceholder);
  });
});
