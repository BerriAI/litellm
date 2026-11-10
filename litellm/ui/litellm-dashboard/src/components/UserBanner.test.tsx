import { renderWithProviders, screen } from "../../tests/test-utils";
import { fireEvent } from "@testing-library/react";
import { vi } from "vitest";
import { UserBanner } from "./UserBanner";
import { UserBanner as UserBannerData } from "./networking";

vi.mock("@/app/(dashboard)/hooks/userBanner/useUserBanner", () => ({
  useUserBanner: vi.fn(),
}));

import { useUserBanner } from "@/app/(dashboard)/hooks/userBanner/useUserBanner";

const publishedBanner: UserBannerData = {
  enabled: true,
  message: "**Maintenance** tonight at 10 PM UTC. See [status page](https://status.example.com).",
  severity: "warning",
  revision: "rev-a",
};

const mockBanner = (banner: UserBannerData | undefined) => {
  vi.mocked(useUserBanner).mockReturnValue({ data: banner } as any);
};

describe("UserBanner", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("renders nothing while the banner has not loaded", () => {
    mockBanner(undefined);
    const { container } = renderWithProviders(<UserBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the banner is unpublished", () => {
    mockBanner({ ...publishedBanner, enabled: false });
    const { container } = renderWithProviders(<UserBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the message is blank", () => {
    mockBanner({ ...publishedBanner, message: "   " });
    const { container } = renderWithProviders(<UserBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the message as markdown with safe external links", () => {
    mockBanner(publishedBanner);
    renderWithProviders(<UserBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Maintenance")).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "status page" });
    expect(link).toHaveAttribute("href", "https://status.example.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it.each([
    ["info", "lucide-info"],
    ["warning", "lucide-triangle-alert"],
    ["error", "lucide-circle-alert"],
  ] as const)("renders the %s severity icon", (severity, iconClass) => {
    mockBanner({ ...publishedBanner, severity });
    const { container } = renderWithProviders(<UserBanner accessToken="token" />);
    expect(container.querySelector(`.${iconClass}`)).toBeInTheDocument();
  });

  it("hides the banner when dismissed and stays hidden for the same content", () => {
    mockBanner(publishedBanner);
    const first = renderWithProviders(<UserBanner accessToken="token" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss banner" }));
    expect(first.container).toBeEmptyDOMElement();

    first.unmount();
    const second = renderWithProviders(<UserBanner accessToken="token" />);
    expect(second.container).toBeEmptyDOMElement();
  });

  it("reappears when the banner content changes after a dismissal", () => {
    mockBanner(publishedBanner);
    const first = renderWithProviders(<UserBanner accessToken="token" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss banner" }));
    first.unmount();

    mockBanner({ ...publishedBanner, message: "All clear, maintenance is done.", revision: "rev-b" });
    renderWithProviders(<UserBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("All clear, maintenance is done.")).toBeInTheDocument();
  });

  it("reappears when identical content is republished under a new revision", () => {
    mockBanner(publishedBanner);
    const first = renderWithProviders(<UserBanner accessToken="token" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss banner" }));
    first.unmount();

    mockBanner({ ...publishedBanner, revision: "rev-b" });
    renderWithProviders(<UserBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });
});
