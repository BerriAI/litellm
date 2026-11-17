/* @vitest-environment jsdom */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { BadgeLink } from "./BadgeLink";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

describe("BadgeLink", () => {
  beforeEach(() => {
    push.mockClear();
  });

  it("renders an anchor pointing at the target href", () => {
    render(<BadgeLink href="/ui/teams?team=t1">My Team</BadgeLink>);
    expect(screen.getByRole("link", { name: "My Team" })).toHaveAttribute("href", "/ui/teams?team=t1");
  });

  it("navigates client-side on plain click", async () => {
    const user = userEvent.setup();
    render(<BadgeLink href="/ui/teams?team=t1">My Team</BadgeLink>);
    await user.click(screen.getByRole("link", { name: "My Team" }));
    expect(push).toHaveBeenCalledWith("/ui/teams?team=t1");
  });

  it("leaves modified clicks to the browser so new-tab shortcuts keep working", async () => {
    const user = userEvent.setup();
    render(<BadgeLink href="/ui/teams?team=t1">My Team</BadgeLink>);
    await user.keyboard("{Meta>}");
    await user.click(screen.getByRole("link", { name: "My Team" }));
    await user.keyboard("{/Meta}");
    expect(push).not.toHaveBeenCalled();
  });

  it("renders a plain same-sized badge when no href is given", () => {
    render(<BadgeLink>all-proxy-models</BadgeLink>);
    expect(screen.getByText("all-proxy-models")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "all-proxy-models" })).not.toBeInTheDocument();
  });
});
