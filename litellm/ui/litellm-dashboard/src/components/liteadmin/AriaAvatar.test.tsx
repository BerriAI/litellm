import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { setServerRootPath } from "@/lib/serverRootPath";
import { AriaAvatar } from "./AriaAvatar";

describe("AriaAvatar", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it.each(["/", "/litellm"])("renders a decorative icon without an image request under server root %s", (root) => {
    setServerRootPath(root);

    render(<AriaAvatar />);
    expect(screen.getByRole("presentation", { hidden: true })).toHaveAttribute("aria-hidden", "true");

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
