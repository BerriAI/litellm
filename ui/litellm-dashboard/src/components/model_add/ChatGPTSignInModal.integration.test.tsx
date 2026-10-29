import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ChatGPTSignInModal from "./ChatGPTSignInModal";

const networking = vi.hoisted(() => ({
  chatGPTLoginStartCall: vi.fn(),
  chatGPTLoginPollCall: vi.fn(),
}));

vi.mock("@/components/networking", () => networking);
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "session-token" }) }));

const START_RESPONSE = {
  user_code: "WXYZ-9876",
  verification_uri: "https://auth.openai.com/codex/device",
  expires_in: 900,
  interval: 5,
  flow_handle: "fh-1",
};

const REQUEST = { ownerTeamId: "team-1", ownerKeyHash: null, displayName: "Shared ChatGPT" };

const renderModal = () => {
  const props = { onClose: vi.fn(), onConnected: vi.fn() };
  render(<ChatGPTSignInModal credentialName="team chatgpt" request={REQUEST} {...props} />);
  return props;
};

const advance = async (seconds: number) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(seconds * 1000);
  });
};

describe("ChatGPTSignInModal", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    networking.chatGPTLoginStartCall.mockReset().mockResolvedValue(START_RESPONSE);
    networking.chatGPTLoginPollCall.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the code and link, polls at the returned interval, and calls onConnected once connected", async () => {
    networking.chatGPTLoginPollCall
      .mockResolvedValueOnce({ status: "pending", account_id: null })
      .mockResolvedValueOnce({ status: "connected", account_id: "acct-1" });
    const props = renderModal();
    await advance(0);

    expect(networking.chatGPTLoginStartCall).toHaveBeenCalledWith("session-token", "team chatgpt", REQUEST);
    expect(screen.getByTestId("chatgpt-user-code")).toHaveTextContent("WXYZ-9876");
    expect(screen.getByRole("link", { name: "https://auth.openai.com/codex/device" })).toHaveAttribute(
      "href",
      "https://auth.openai.com/codex/device",
    );

    await advance(4.9);
    expect(networking.chatGPTLoginPollCall).not.toHaveBeenCalled();
    await advance(0.1);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(1);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledWith("session-token", "team chatgpt", "fh-1");
    expect(props.onConnected).not.toHaveBeenCalled();

    await advance(5);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(2);
    expect(props.onConnected).toHaveBeenCalledTimes(1);

    await advance(30);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(2);
  });

  it("shows the error from a rejected poll, stops polling, and starts a new flow on request", async () => {
    networking.chatGPTLoginPollCall.mockRejectedValue(new Error("Invalid or expired flow handle"));
    const props = renderModal();
    await advance(0);
    await advance(5);

    expect(screen.getByRole("alert")).toHaveTextContent("Invalid or expired flow handle");
    expect(props.onConnected).not.toHaveBeenCalled();
    await advance(30);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "Start again" }));
    await advance(0);
    expect(networking.chatGPTLoginStartCall).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("chatgpt-user-code")).toBeInTheDocument();
  });

  it("stops polling and says the code expired once its lifetime passes", async () => {
    networking.chatGPTLoginStartCall.mockResolvedValue({ ...START_RESPONSE, expires_in: 12 });
    networking.chatGPTLoginPollCall.mockResolvedValue({ status: "pending", account_id: null });
    const props = renderModal();
    await advance(0);
    await advance(5);
    await advance(5);
    await advance(5);

    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("alert")).toHaveTextContent("The sign-in code expired");
    await advance(30);
    expect(networking.chatGPTLoginPollCall).toHaveBeenCalledTimes(2);
    expect(props.onConnected).not.toHaveBeenCalled();
  });

  it("shows the proxy error when the start request fails", async () => {
    networking.chatGPTLoginStartCall.mockRejectedValue(new Error("Team id does not exist"));
    renderModal();
    await advance(0);

    expect(screen.getByRole("alert")).toHaveTextContent("Team id does not exist");
    expect(networking.chatGPTLoginPollCall).not.toHaveBeenCalled();
  });
});
