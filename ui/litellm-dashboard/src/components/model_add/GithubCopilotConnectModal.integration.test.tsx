import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import GithubCopilotConnectModal from "./GithubCopilotConnectModal";

const networking = vi.hoisted(() => ({
  userConnectionStartCall: vi.fn(),
  userConnectionPollCall: vi.fn(),
}));

vi.mock("@/components/networking", () => networking);
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => ({ accessToken: "session-token" }) }));

const START_RESPONSE = {
  user_code: "ABCD-1234",
  verification_uri: "https://github.com/login/device",
  expires_in: 900,
  interval: 5,
  flow_handle: "fh-1",
};

const renderModal = () => {
  const props = { onClose: vi.fn(), onConnected: vi.fn(), onDisconnect: vi.fn() };
  render(<GithubCopilotConnectModal credentialName="copilot-per-user" {...props} />);
  return props;
};

const advance = async (seconds: number) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(seconds * 1000);
  });
};

describe("GithubCopilotConnectModal", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
    networking.userConnectionStartCall.mockReset().mockResolvedValue(START_RESPONSE);
    networking.userConnectionPollCall.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows the device code and link, polls at the given interval, honours slow_down and ends connected", async () => {
    networking.userConnectionPollCall
      .mockResolvedValueOnce({ status: "pending" })
      .mockResolvedValueOnce({ status: "slow_down", interval: 10 })
      .mockResolvedValueOnce({ status: "connected", github_login: "octocat" });
    const props = renderModal();
    await advance(0);

    expect(networking.userConnectionStartCall).toHaveBeenCalledWith("session-token", "copilot-per-user");
    expect(screen.getByTestId("github-device-code")).toHaveTextContent("ABCD-1234");
    expect(screen.getByRole("link", { name: "https://github.com/login/device" })).toHaveAttribute(
      "href",
      "https://github.com/login/device",
    );

    await advance(4.9);
    expect(networking.userConnectionPollCall).not.toHaveBeenCalled();
    await advance(0.1);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(1);
    expect(networking.userConnectionPollCall).toHaveBeenCalledWith("session-token", "copilot-per-user", "fh-1");
    await advance(5);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(2);

    await advance(9.9);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(2);
    await advance(0.1);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(3);

    expect(screen.getByText("Connected as @octocat")).toBeInTheDocument();
    expect(props.onConnected).toHaveBeenCalledTimes(1);
    await advance(30);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(3);

    fireEvent.click(screen.getByRole("button", { name: "Disconnect" }));
    expect(props.onDisconnect).toHaveBeenCalledTimes(1);
  });

  it("stops polling and offers a restart when the device code expires", async () => {
    networking.userConnectionStartCall.mockResolvedValue({ ...START_RESPONSE, expires_in: 12 });
    networking.userConnectionPollCall.mockResolvedValue({ status: "pending" });
    renderModal();
    await advance(0);
    await advance(5);
    await advance(5);
    await advance(5);

    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("alert")).toHaveTextContent("The device code expired");
    await advance(30);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(2);

    fireEvent.click(screen.getByRole("button", { name: "Start again" }));
    await advance(0);
    expect(networking.userConnectionStartCall).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("github-device-code")).toBeInTheDocument();
  });

  it.each([
    ["denied", "denied on GitHub"],
    ["no_copilot_seat", "does not have GitHub Copilot access"],
    ["expired", "The device code expired"],
  ])("shows the %s error and stops polling", async (status, message) => {
    networking.userConnectionPollCall.mockResolvedValue({ status });
    const props = renderModal();
    await advance(0);
    await advance(5);

    expect(screen.getByRole("alert")).toHaveTextContent(message);
    expect(props.onConnected).not.toHaveBeenCalled();
    await advance(30);
    expect(networking.userConnectionPollCall).toHaveBeenCalledTimes(1);
  });

  it("shows the proxy error when the start request fails", async () => {
    networking.userConnectionStartCall.mockRejectedValue(new Error("Credential not found"));
    renderModal();
    await advance(0);

    expect(screen.getByRole("alert")).toHaveTextContent("Credential not found");
    expect(networking.userConnectionPollCall).not.toHaveBeenCalled();
  });
});
