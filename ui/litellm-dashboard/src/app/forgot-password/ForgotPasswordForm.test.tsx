import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ForgotPasswordForm } from "./ForgotPasswordForm";
import * as networking from "@/components/networking";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("ForgotPasswordForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("submits the typed email and shows the generic success message", async () => {
    vi.spyOn(networking, "forgotPasswordCall").mockResolvedValue({ message: "sent" });
    const user = userEvent.setup();
    renderWithClient(<ForgotPasswordForm />);

    await user.type(screen.getByLabelText("Email Address"), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /send reset link/i }));

    await waitFor(() => {
      expect(networking.forgotPasswordCall).toHaveBeenCalledWith("alice@example.com");
    });
    await waitFor(() => {
      expect(
        screen.getByText("If an account exists for this email, a password reset link has been sent."),
      ).toBeInTheDocument();
    });
  });

  it("shows a generic error message when the request fails", async () => {
    vi.spyOn(networking, "forgotPasswordCall").mockRejectedValue(new Error("Too many requests"));
    const user = userEvent.setup();
    renderWithClient(<ForgotPasswordForm />);

    await user.type(screen.getByLabelText("Email Address"), "alice@example.com");
    await user.click(screen.getByRole("button", { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText("Too many requests")).toBeInTheDocument();
    });
  });

  it("has a link back to the login page", () => {
    renderWithClient(<ForgotPasswordForm />);
    expect(screen.getByRole("link", { name: /back to login/i })).toBeInTheDocument();
  });
});
