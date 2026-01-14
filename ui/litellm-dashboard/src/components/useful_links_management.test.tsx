import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Modal } from "antd";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import UsefulLinksManagement from "./useful_links_management";
import NotificationsManager from "./molecules/notifications_manager";
import { getPublicModelHubInfo, updateUsefulLinksCall, getProxyBaseUrl } from "./networking";

vi.mock("./networking", () => ({
  getPublicModelHubInfo: vi.fn(),
  updateUsefulLinksCall: vi.fn(),
  getProxyBaseUrl: vi.fn(),
}));

vi.mock("./molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

const mockedGetPublicModelHubInfo = vi.mocked(getPublicModelHubInfo);
const mockedUpdateUsefulLinksCall = vi.mocked(updateUsefulLinksCall);
const mockedGetProxyBaseUrl = vi.mocked(getProxyBaseUrl);
const mockedNotifications = vi.mocked(NotificationsManager);

let modalSuccessSpy: any;

describe("UsefulLinksManagement", () => {
  beforeEach(() => {
    mockedGetPublicModelHubInfo.mockResolvedValue({
      docs_title: "Docs",
      custom_docs_description: null,
      litellm_version: "1.0.0",
      useful_links: {},
    });
    mockedUpdateUsefulLinksCall.mockResolvedValue({});
    mockedGetProxyBaseUrl.mockReturnValue("https://proxy.example.com");
    modalSuccessSpy = vi.spyOn(Modal, "success").mockImplementation(() => ({ destroy: vi.fn() }) as any);
  });

  afterEach(() => {
    modalSuccessSpy.mockRestore();
    vi.clearAllMocks();
  });

  it("should render link management for admin users", async () => {
    render(<UsefulLinksManagement accessToken="token" userRole="Admin" />);

    expect(await screen.findByText("Link Management")).toBeInTheDocument();
    await waitFor(() => expect(mockedGetPublicModelHubInfo).toHaveBeenCalled());
  });

  it("should add a new link when fields are valid", async () => {
    const user = userEvent.setup();
    render(<UsefulLinksManagement accessToken="token" userRole="Admin" />);

    const displayNameInput = await screen.findByPlaceholderText("Friendly name");
    const urlInput = screen.getByPlaceholderText("https://example.com");

    await user.type(displayNameInput, "Docs");
    await user.type(urlInput, "https://docs.example.com");
    await user.click(screen.getByRole("button", { name: /add link/i }));

    await waitFor(() =>
      expect(mockedUpdateUsefulLinksCall).toHaveBeenCalledWith("token", {
        Docs: { url: "https://docs.example.com", index: 0 },
      }),
    );

    expect(await screen.findByText("Docs")).toBeInTheDocument();
    expect(screen.getByText("https://docs.example.com")).toBeInTheDocument();
    expect(mockedNotifications.success).toHaveBeenCalledWith("Link added successfully");
  });

  it("should rearrange links and save the new order", async () => {
    const user = userEvent.setup();
    mockedGetPublicModelHubInfo.mockResolvedValue({
      docs_title: "Docs",
      custom_docs_description: null,
      litellm_version: "1.0.0",
      useful_links: {
        "First Link": "https://first.example.com",
        "Second Link": "https://second.example.com",
        "Third Link": "https://third.example.com",
      },
    });

    render(<UsefulLinksManagement accessToken="token" userRole="Admin" />);

    await waitFor(() => expect(screen.getByText("First Link")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /rearrange order/i }));

    const secondLinkMoveUpButton = screen.getByTestId("move-up-1-Second Link");
    await user.click(secondLinkMoveUpButton);

    await user.click(screen.getByRole("button", { name: /save order/i }));

    await waitFor(() =>
      expect(mockedUpdateUsefulLinksCall).toHaveBeenCalledWith("token", {
        "Second Link": { url: "https://second.example.com", index: 0 },
        "First Link": { url: "https://first.example.com", index: 1 },
        "Third Link": { url: "https://third.example.com", index: 2 },
      }),
    );

    expect(mockedNotifications.success).toHaveBeenCalledWith("Link order saved successfully");
  });
});
