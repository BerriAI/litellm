import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import BulkCreateUsersButton from "./bulk_create_users_button";
import { getProxyUISettings } from "./networking";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("BulkCreateUsersButton", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.restoreAllMocks();
  });

  const renderBulkCreateUsersButton = async () => {
    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    await waitFor(() => expect(getProxyUISettings).toHaveBeenCalled());
  };

  it("should render", async () => {
    await renderBulkCreateUsersButton();
    expect(screen.getByRole("button", { name: "+ Bulk Invite Users" })).toBeInTheDocument();
  });

  it("should download the CSV template when the template button is clicked", async () => {
    if (!URL.createObjectURL) {
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        value: vi.fn(),
      });
    }

    const createObjectURLSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:template");
    const revokeObjectURLSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const anchorClickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click");

    await renderBulkCreateUsersButton();

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "+ Bulk Invite Users" }));
    });

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /Download CSV Template/ }));
    });

    expect(createObjectURLSpy).toHaveBeenCalledWith(expect.any(Blob));
    expect(anchorClickSpy).toHaveBeenCalledOnce();
    expect(revokeObjectURLSpy).toHaveBeenCalledWith("blob:template");
  });
});
