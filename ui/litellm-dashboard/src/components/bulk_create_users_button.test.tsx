import { fireEvent, render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import BulkCreateUsersButton from "./bulk_create_users_button";

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
  it("should render", () => {
    const { getByText } = render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });

  it("downloads the CSV template when clicked", async () => {
    const originalCreateObjectURL = window.URL.createObjectURL;
    const originalRevokeObjectURL = window.URL.revokeObjectURL;
    const createObjectURLSpy = vi.fn(() => "blob:test");
    const revokeObjectURLSpy = vi.fn();
    const appendChildSpy = vi.spyOn(document.body, "appendChild");
    const removeChildSpy = vi.spyOn(document.body, "removeChild");
    const clickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    let createdAnchor: HTMLAnchorElement | null = null;

    window.URL.createObjectURL = createObjectURLSpy;
    window.URL.revokeObjectURL = revokeObjectURLSpy;

    const createElementSpy = vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
      if (tagName === "a") {
        const anchor = originalCreateElement("a");
        anchor.click = clickSpy;
        createdAnchor = anchor;
        return anchor;
      }

      return originalCreateElement(tagName);
    });

    const { getByText, findByText } = render(
      <BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />
    );

    fireEvent.click(getByText("+ Bulk Invite Users"));
    fireEvent.click(await findByText("Download CSV Template"));

    expect(createObjectURLSpy).toHaveBeenCalledOnce();
    expect(clickSpy).toHaveBeenCalledOnce();
    expect(createdAnchor).not.toBeNull();
    expect(appendChildSpy).toHaveBeenCalledWith(createdAnchor);
    expect(removeChildSpy).toHaveBeenCalledWith(createdAnchor);
    expect(revokeObjectURLSpy).toHaveBeenCalledWith("blob:test");

    appendChildSpy.mockRestore();
    removeChildSpy.mockRestore();
    createElementSpy.mockRestore();
    window.URL.createObjectURL = originalCreateObjectURL;
    window.URL.revokeObjectURL = originalRevokeObjectURL;
  });
});
