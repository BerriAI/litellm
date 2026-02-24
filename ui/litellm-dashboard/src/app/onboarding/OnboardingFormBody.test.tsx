import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { OnboardingFormBody } from "./OnboardingFormBody";

const defaultProps = {
  variant: "signup" as const,
  userEmail: "test@example.com",
  isPending: false,
  claimError: null,
  onSubmit: vi.fn(),
};

describe("OnboardingFormBody", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows 'Sign Up' heading for signup variant", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByRole("heading", { name: "Sign Up" })).toBeInTheDocument();
  });

  it("shows 'Reset Password' heading for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(screen.getByRole("heading", { name: "Reset Password" })).toBeInTheDocument();
  });

  it("shows SSO alert for signup variant", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByText("SSO")).toBeInTheDocument();
  });

  it("hides SSO alert for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(screen.queryByText("SSO")).not.toBeInTheDocument();
  });

  it("pre-fills the email field with userEmail", async () => {
    render(<OnboardingFormBody {...defaultProps} userEmail="user@example.com" />);
    await waitFor(() => {
      expect(screen.getByLabelText("Email Address")).toHaveValue("user@example.com");
    });
  });

  it("disables the email field", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByLabelText("Email Address")).toBeDisabled();
  });

  it("shows claimError message when claimError is set", () => {
    render(<OnboardingFormBody {...defaultProps} claimError="Something went wrong" />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("does not show claimError message when claimError is null", () => {
    render(<OnboardingFormBody {...defaultProps} claimError={null} />);
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  it("shows a loading indicator on the submit button when isPending is true", () => {
    render(<OnboardingFormBody {...defaultProps} isPending={true} />);
    // antd Button with loading={true} adds .ant-btn-loading class
    expect(document.querySelector(".ant-btn-loading")).toBeInTheDocument();
  });

  it("calls onSubmit with the typed password on form submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Password"), "mypassword");
    await user.click(screen.getByRole("button", { name: /sign up/i }));

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({ password: "mypassword" })
      );
    });
  });

  it("shows 'Reset Password' on the submit button for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(
      screen.getByRole("button", { name: /reset password/i })
    ).toBeInTheDocument();
  });
});
