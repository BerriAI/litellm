import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import { GuardrailConfig } from "./GuardrailConfig";

describe("GuardrailConfig", () => {
  const defaultProps = {
    guardrailName: "Content Safety",
    guardrailType: "Content Safety",
    provider: "bedrock",
  };

  it("should render", () => {
    render(<GuardrailConfig {...defaultProps} />);
    expect(screen.getByText("Parameters")).toBeInTheDocument();
  });

  it("should display the guardrail name in the parameters description", () => {
    render(<GuardrailConfig {...defaultProps} />);
    expect(screen.getByText(/Configure Content Safety behavior/)).toBeInTheDocument();
  });

  it("should show version history when 'View history' is clicked", async () => {
    const user = userEvent.setup();
    render(<GuardrailConfig {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /view history/i }));
    expect(screen.getByText("Initial configuration")).toBeInTheDocument();
    expect(screen.getByText("Added custom categories list")).toBeInTheDocument();
  });

  it("should toggle version history text between View/Hide", async () => {
    const user = userEvent.setup();
    render(<GuardrailConfig {...defaultProps} />);
    const button = screen.getByRole("button", { name: /view history/i });
    await user.click(button);
    expect(screen.getByRole("button", { name: /hide history/i })).toBeInTheDocument();
  });

  it("should show custom code textarea when custom code override is toggled on", async () => {
    const user = userEvent.setup();
    render(<GuardrailConfig {...defaultProps} />);
    const switches = screen.getAllByRole("switch");
    // The second switch is the custom code override toggle
    const customCodeSwitch = switches[1];
    await user.click(customCodeSwitch);
    expect(screen.getByPlaceholderText(/async def evaluate/)).toBeInTheDocument();
  });

  it("should hide custom code textarea when custom code override is off", () => {
    render(<GuardrailConfig {...defaultProps} />);
    // There's an input for categories, but no textarea
    expect(screen.queryByPlaceholderText(/async def evaluate/)).not.toBeInTheDocument();
  });

  it("should show the re-run button in idle state", () => {
    render(<GuardrailConfig {...defaultProps} />);
    expect(screen.getByRole("button", { name: /re-run on failing logs/i })).toBeInTheDocument();
  });

  it("should show loading state when re-run is clicked", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<GuardrailConfig {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /re-run on failing logs/i }));
    expect(screen.getByText(/Running on 10 samples/)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("should show success message after re-run completes", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<GuardrailConfig {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /re-run on failing logs/i }));
    act(() => { vi.advanceTimersByTime(2500); });
    expect(screen.getByText(/7\/10 would now pass/)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("should display the Revert and Save buttons", () => {
    render(<GuardrailConfig {...defaultProps} />);
    expect(screen.getByRole("button", { name: /revert/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /save as v4/i })).toBeInTheDocument();
  });
});
