import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import ObservedConnections from "./ObservedConnections";

describe("observed ROI connection preview", () => {
  it.each([
    ["GitHub", "app"],
    ["GitHub", "token"],
    ["GitLab", "app"],
    ["GitLab", "token"],
  ] as const)("walks through %s via %s without claiming a real connection", async (provider, method) => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ObservedConnections repo="example/service" onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: provider, exact: true }));
    if (provider === "GitLab") {
      fireEvent.change(screen.getByLabelText("GitLab instance"), { target: { value: "https://git.example.test" } });
      expect(screen.getByLabelText("GitLab instance")).toHaveValue("https://git.example.test");
    }
    if (method === "token") await user.click(screen.getByRole("button", { name: /Use an access token/ }));
    await user.click(
      screen.getByRole("button", {
        name: method === "app" ? `Preview ${provider} authorization` : "Preview token setup",
      }),
    );
    expect(screen.getByText(new RegExp(`This preview does not contact ${provider}`))).toBeInTheDocument();
    if (method === "token") expect(screen.getByLabelText("Access token")).toHaveAttribute("readonly");
    await user.click(screen.getByRole("button", { name: "Continue with example" }));
    await user.click(screen.getByRole("checkbox"));
    expect(screen.getByRole("button", { name: "Preview finish" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Preview finish" }));
    expect(screen.getByText("Preview complete. No new connection was created")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back to report" }));
    expect(onClose).toHaveBeenCalledOnce();
  });
});
