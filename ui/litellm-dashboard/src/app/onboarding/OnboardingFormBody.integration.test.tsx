import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OnboardingFormBody } from "./OnboardingFormBody";

const defaultProps = {
  variant: "signup" as const,
  userEmail: "test@example.com",
  isPending: false,
  claimError: null,
};

describe("OnboardingFormBody submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends exactly {password} and never leaks the email field", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "hunter2" } });
    await user.click(screen.getByRole("button", { name: "Sign Up" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({ password: "hunter2" });
  });

  it("submits on Enter from the password field", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Password"), "hunter2{Enter}");

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({ password: "hunter2" });
  });

  it("blocks submit and marks the password invalid when it is empty", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Sign Up" }));

    await waitFor(() => expect(screen.getByLabelText("Password")).toHaveAttribute("aria-invalid", "true"));
    expect(screen.getByText("Create a password for your account")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps the password help text always visible and variant specific", async () => {
    const { unmount } = render(<OnboardingFormBody {...defaultProps} onSubmit={vi.fn()} />);
    expect(screen.getByText("Create a password for your account")).toBeInTheDocument();
    unmount();

    render(<OnboardingFormBody {...defaultProps} variant="reset_password" onSubmit={vi.fn()} />);
    expect(screen.getByText("Enter your new password")).toBeInTheDocument();
  });
});
