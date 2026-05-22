import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { NotificationsBell, AGENT_PLATFORM_URL } from "./NotificationsBell";
import React from "react";
import userEvent from "@testing-library/user-event";

describe("NotificationsBell", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("should open notifications with Agent Platform details and GitHub link", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBell />);
    await user.click(screen.getByRole("button", { name: /^notifications$/i }));
    expect(screen.getByText(/LiteLLM Agent Platform/i)).toBeInTheDocument();
    const githubBtn = screen.getByRole("link", { name: /^GitHub$/i });
    expect(githubBtn).toHaveAttribute("href", AGENT_PLATFORM_URL);
    expect(githubBtn).toHaveAttribute("target", "_blank");
    expect(githubBtn).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should offer mark as read when announcement is unread", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBell />);
    await user.click(screen.getByRole("button", { name: /^notifications$/i }));
    expect(screen.getByRole("button", { name: /^mark as read$/i })).toBeInTheDocument();
  });

  it("should hide mark as read and persist after marking read", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBell />);
    await user.click(screen.getByRole("button", { name: /^notifications$/i }));
    await user.click(screen.getByRole("button", { name: /^mark as read$/i }));
    expect(localStorage.getItem("litellmHideAgentPlatformBanner")).toBe("true");
    await user.click(screen.getByRole("button", { name: /^notifications$/i }));
    expect(screen.queryByRole("button", { name: /^mark as read$/i })).not.toBeInTheDocument();
  });

  it("should not show mark as read when previously dismissed", async () => {
    localStorage.setItem("litellmHideAgentPlatformBanner", "true");
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBell />);
    await user.click(screen.getByRole("button", { name: /^notifications$/i }));
    expect(screen.queryByRole("button", { name: /^mark as read$/i })).not.toBeInTheDocument();
  });
});
