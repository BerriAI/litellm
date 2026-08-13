import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import RoutePreview from "./route_preview";

vi.mock("./networking", () => ({ getProxyBaseUrl: () => "http://proxy.test" }));

describe("RoutePreview", () => {
  it("stays hidden until both route values are provided", () => {
    render(<RoutePreview pathValue="/images" targetValue="" includeSubpath={false} />);

    expect(screen.queryByText("Route Preview")).not.toBeInTheDocument();
  });

  it("shows the proxy route and forwarding target", () => {
    render(<RoutePreview pathValue="/images" targetValue="https://upstream.test" includeSubpath={false} />);

    expect(screen.getByText("http://proxy.test/images")).toBeInTheDocument();
    expect(screen.getByText("https://upstream.test")).toBeInTheDocument();
    expect(screen.getByText(/Not seeing the routing you wanted/)).toBeInTheDocument();
  });

  it("previews appended subpaths when enabled", () => {
    render(<RoutePreview pathValue="/images" targetValue="https://upstream.test" includeSubpath />);

    expect(screen.getByText("With subpaths:")).toBeInTheDocument();
    expect(screen.getAllByText("/v1/text-to-image/base/model")).toHaveLength(2);
    expect(screen.getByText(/Any path after \/images will be appended/)).toBeInTheDocument();
  });
});
