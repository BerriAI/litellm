import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("./ResetPasswordForm", () => ({
  ResetPasswordForm: ({ token }: { token: string | null }) => <div>token:{token ?? "none"}</div>,
}));

import ResetPassword from "./page";

describe("ResetPassword page", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/ui/reset-password/");
  });

  it("reads the token from the URL fragment, not a query param", async () => {
    window.history.replaceState(null, "", "/ui/reset-password/#token=abc123");

    render(<ResetPassword />);

    await waitFor(() => {
      expect(screen.getByText("token:abc123")).toBeInTheDocument();
    });
  });

  it("passes null when there is no fragment token", async () => {
    render(<ResetPassword />);

    await waitFor(() => {
      expect(screen.getByText("token:none")).toBeInTheDocument();
    });
  });

  it("strips the token from the visible URL after reading it", async () => {
    window.history.replaceState(null, "", "/ui/reset-password/#token=abc123");

    render(<ResetPassword />);

    await waitFor(() => {
      expect(window.location.hash).toBe("");
    });
    expect(window.location.pathname).toBe("/ui/reset-password/");
  });
});
