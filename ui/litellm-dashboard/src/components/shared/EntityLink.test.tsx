/* @vitest-environment jsdom */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EntityLink } from "./EntityLink";

const push = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

describe("EntityLink", () => {
  beforeEach(() => {
    push.mockClear();
  });

  it("renders an anchor pointing at the target href", () => {
    render(<EntityLink href="/ui/users?user=u1">alice</EntityLink>);
    expect(screen.getByRole("link", { name: "alice" })).toHaveAttribute("href", "/ui/users?user=u1");
  });

  it("navigates client-side on plain click", async () => {
    const user = userEvent.setup();
    render(<EntityLink href="/ui/users?user=u1">alice</EntityLink>);
    await user.click(screen.getByRole("link", { name: "alice" }));
    expect(push).toHaveBeenCalledWith("/ui/users?user=u1");
  });

  it("leaves modified clicks to the browser so new-tab shortcuts keep working", async () => {
    const user = userEvent.setup();
    render(<EntityLink href="/ui/users?user=u1">alice</EntityLink>);
    await user.keyboard("{Meta>}");
    await user.click(screen.getByRole("link", { name: "alice" }));
    await user.keyboard("{/Meta}");
    expect(push).not.toHaveBeenCalled();
  });
});
