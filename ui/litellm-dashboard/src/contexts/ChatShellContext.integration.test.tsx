import { screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { ChatShellProvider, useChatShell } from "./ChatShellContext";

function ActiveConversationId() {
  const { activeConversationId } = useChatShell();
  return <span>{activeConversationId}</span>;
}

describe("ChatShellProvider URL state", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("reads the active conversation ID from the nuqs adapter", () => {
    renderWithProviders(
      <ChatShellProvider
        accessToken="token"
        userId="test-user"
        userEmail="test@example.com"
        userRole="Admin"
        premiumUser={false}
      >
        <ActiveConversationId />
      </ChatShellProvider>,
      { searchParams: "?id=conversation-123" },
    );

    expect(screen.getByText("conversation-123")).toBeInTheDocument();
  });
});
