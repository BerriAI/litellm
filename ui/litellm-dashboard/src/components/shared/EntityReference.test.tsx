/* @vitest-environment jsdom */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import { EntityReference, UserReference } from "./EntityReference";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

describe("EntityReference", () => {
  it("shows the name as the primary linked text when a name and href exist", () => {
    render(<EntityReference id="team-123" name="Review Team" href="/ui/teams?team=team-123" />);
    expect(screen.getByRole("link", { name: "Review Team" })).toHaveAttribute("href", "/ui/teams?team=team-123");
    expect(screen.queryByText("team-123")).not.toBeInTheDocument();
  });

  it("keeps the raw id reachable as a tooltip when a name is shown", async () => {
    const user = userEvent.setup();
    render(<EntityReference id="team-123" name="Review Team" />);
    await user.hover(screen.getByText("Review Team"));
    expect(await screen.findByText("team-123")).toBeInTheDocument();
  });

  it("caps the named trigger at the container width so long names truncate", () => {
    const { container } = render(<EntityReference id="team-123" name="Review Team" />);
    expect(container.querySelector("span")).toHaveClass("max-w-full");
  });

  it("falls back to the raw id in mono when there is no name", () => {
    render(<EntityReference id="team-123" href="/ui/teams?team=team-123" />);
    const link = screen.getByRole("link", { name: "team-123" });
    expect(link).toHaveAttribute("href", "/ui/teams?team=team-123");
    expect(link).toHaveClass("font-mono");
  });

  it("renders the raw id without a link when neither name nor href exist", () => {
    render(<EntityReference id="team-123" />);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("team-123")).toBeInTheDocument();
  });
});

describe("UserReference", () => {
  it("renders nothing for a nullish user id", () => {
    const { container } = render(<UserReference userId={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the display name linked to the user detail page", () => {
    render(<UserReference userId="user-9" displayName="Ada Reviewer" />);
    const link = screen.getByRole("link", { name: "Ada Reviewer" });
    expect(link).toHaveAttribute("href", expect.stringContaining("user-9"));
  });

  it("keeps the default proxy admin badge when the sentinel id has no name", () => {
    render(<UserReference userId={DEFAULT_PROXY_ADMIN_USER_ID} />);
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });

  it("falls back to the raw user id when no display name resolved", () => {
    render(<UserReference userId="user-9" />);
    expect(screen.getByRole("link", { name: "user-9" })).toBeInTheDocument();
  });
});
