import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { OnboardingFormBody } from "./OnboardingFormBody";

const translationState = vi.hoisted(() => ({ language: "en" as "en" | "ru" }));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources[translationState.language].auth) ?? key,
    }),
  };
});

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
    translationState.language = "en";
  });

  it("renders the signup form in Russian", () => {
    translationState.language = "ru";

    render(<OnboardingFormBody {...defaultProps} />);

    expect(screen.getByRole("heading", { name: "Регистрация" })).toBeInTheDocument();
    expect(screen.getByLabelText("Эл. почта")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Зарегистрироваться" })).toBeInTheDocument();
  });

  it("should show 'Sign Up' heading for signup variant", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByRole("heading", { name: "Sign Up" })).toBeInTheDocument();
  });

  it("should show 'Reset Password' heading for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(screen.getByRole("heading", { name: "Reset Password" })).toBeInTheDocument();
  });

  it("should show SSO alert for signup variant", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByText("SSO")).toBeInTheDocument();
  });

  it("should hide SSO alert for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(screen.queryByText("SSO")).not.toBeInTheDocument();
  });

  it("should pre-fill the email field with userEmail", async () => {
    render(<OnboardingFormBody {...defaultProps} userEmail="user@example.com" />);
    await waitFor(() => {
      expect(screen.getByLabelText("Email Address")).toHaveValue("user@example.com");
    });
  });

  it("should disable the email field", () => {
    render(<OnboardingFormBody {...defaultProps} />);
    expect(screen.getByLabelText("Email Address")).toBeDisabled();
  });

  it("should show claimError message when claimError is set", () => {
    render(<OnboardingFormBody {...defaultProps} claimError="Something went wrong" />);
    expect(screen.getByText("Something went wrong")).toBeInTheDocument();
  });

  it("should not show claimError message when claimError is null", () => {
    render(<OnboardingFormBody {...defaultProps} claimError={null} />);
    expect(screen.queryByText("Something went wrong")).not.toBeInTheDocument();
  });

  it("should show a loading indicator on the submit button when isPending is true", () => {
    render(<OnboardingFormBody {...defaultProps} isPending={true} />);
    // antd v5 renders a loading icon with aria-label="loading" inside the button
    expect(screen.getByRole("img", { name: "loading" })).toBeInTheDocument();
  });

  it("should call onSubmit with the typed password on form submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Password"), "mypassword");
    await user.click(screen.getByRole("button", { name: /sign up/i }));

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({ password: "mypassword" }));
    });
  });

  it("should show 'Reset Password' on the submit button for reset_password variant", () => {
    render(<OnboardingFormBody {...defaultProps} variant="reset_password" />);
    expect(screen.getByRole("button", { name: /reset password/i })).toBeInTheDocument();
  });
});
