import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import CacheControlInjectionPoints, { type CacheControlInjectionPoint } from "./cache_control_settings";

const ROLE_HINT = "LiteLLM will mark all messages of this role as cacheable";
const INDEX_HINT = "(Optional) If set litellm will mark the message at this index as cacheable";

const ONE_POINT: CacheControlInjectionPoint[] = [{ location: "message" }];

const tabTo = async (user: ReturnType<typeof userEvent.setup>, name: string): Promise<void> => {
  for (let step = 0; step < 8; step++) {
    await user.tab();
    if (document.activeElement === screen.getByRole("button", { name })) {
      return;
    }
  }
  throw new Error(`${name} is not reachable by keyboard`);
};

describe("CacheControlInjectionPoints field hints", () => {
  it("explains on the Role field that the role marks every message of that role cacheable", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await user.hover(screen.getByRole("button", { name: "Role help" }));

    expect(await screen.findByText(ROLE_HINT)).toBeInTheDocument();
  });

  it("explains on the Index field that it is optional and marks that message cacheable", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await user.hover(screen.getByRole("button", { name: "Index help" }));

    expect(await screen.findByText(INDEX_HINT)).toBeInTheDocument();
  });

  it("reveals the Role hint on keyboard focus, so it is reachable without a pointer", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await tabTo(user, "Role help");

    expect(await screen.findByText(ROLE_HINT)).toBeInTheDocument();
  });

  it("reveals the Index hint on keyboard focus, so it is reachable without a pointer", async () => {
    const user = userEvent.setup();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    await tabTo(user, "Index help");

    expect(await screen.findByText(INDEX_HINT)).toBeInTheDocument();
  });

  it("keeps both hints behind a hover or a focus rather than rendering them inline", () => {
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={vi.fn()} />);

    expect(screen.queryByText(ROLE_HINT)).not.toBeInTheDocument();
    expect(screen.queryByText(INDEX_HINT)).not.toBeInTheDocument();
  });

  it("gives every row its own pair of hints", () => {
    render(
      <CacheControlInjectionPoints value={[{ location: "message" }, { location: "message" }]} onChange={vi.fn()} />,
    );

    expect(screen.getAllByRole("button", { name: "Role help" })).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: "Index help" })).toHaveLength(2);
  });

  it("still reports a typed index as a string through onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<CacheControlInjectionPoints value={ONE_POINT} onChange={onChange} />);

    await user.type(screen.getByPlaceholderText("Optional"), "3");

    expect(onChange).toHaveBeenLastCalledWith([{ location: "message", index: "3" }]);
  });
});
