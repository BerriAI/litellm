import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ResetPasswordForm } from "./ResetPasswordForm";
import * as networking from "@/components/networking";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("ResetPasswordForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows an invalid-link message when there is no token", () => {
    renderWithClient(<ResetPasswordForm token={null} />);
    expect(screen.getByText("This link is invalid or has expired.")).toBeInTheDocument();
  });

  it("shows an invalid-link message when validation fails", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockRejectedValue(new Error("invalid"));
    renderWithClient(<ResetPasswordForm token="bad-token" />);
    await waitFor(() => {
      expect(screen.getByText("This link is invalid or has expired.")).toBeInTheDocument();
    });
  });

  it("shows the target email and submits matching passwords", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    vi.spyOn(networking, "resetPasswordCall").mockResolvedValue({ message: "ok" });
    const user = userEvent.setup();
    renderWithClient(<ResetPasswordForm token="good-token" />);

    await waitFor(() => {
      expect(screen.getByText(/alice@example.com/)).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText("New Password"), "correct horse battery staple");
    await user.type(screen.getByLabelText("Confirm New Password"), "correct horse battery staple");
    await user.click(screen.getByRole("button", { name: /reset password/i }));

    await waitFor(() => {
      expect(networking.resetPasswordCall).toHaveBeenCalledWith("good-token", "correct horse battery staple");
    });
    await waitFor(() => {
      expect(screen.getByText("Password reset successfully.")).toBeInTheDocument();
    });
  });

  it("shows a validation error when the two password fields do not match", async () => {
    vi.spyOn(networking, "validateResetTokenCall").mockResolvedValue({ user_email: "alice@example.com" });
    const user = userEvent.setup();
    renderWithClient(<ResetPasswordForm token="good-token" />);

    await waitFor(() => screen.getByLabelText("New Password"));
    await user.type(screen.getByLabelText("New Password"), "password-one");
    await user.type(screen.getByLabelText("Confirm New Password"), "password-two");
    await user.click(screen.getByRole("button", { name: /reset password/i }));

    await waitFor(() => {
      expect(screen.getByText("Passwords do not match")).toBeInTheDocument();
    });
    expect(networking.resetPasswordCall).not.toHaveBeenCalled();
  });
});
