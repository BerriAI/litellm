import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { setServerRootPath } from "@/lib/serverRootPath";
import { AriaAvatar } from "./AriaAvatar";

describe("AriaAvatar", () => {
  afterEach(() => {
    setServerRootPath("/");
  });

  it.each(["/", "/litellm"])("loads the bundled image when the server root is %s", (root) => {
    setServerRootPath(root);

    render(<AriaAvatar />);

    expect(screen.getByRole("presentation", { hidden: true })).toHaveAttribute(
      "src",
      "/_next/static/media/aria-agent.png",
    );
  });
});
