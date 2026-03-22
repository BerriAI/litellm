import { render, screen } from "@testing-library/react";
import RoutePreview from "./route_preview";

vi.mock("./networking", () => ({
  getProxyBaseUrl: () => "https://proxy.example.com",
}));

describe("RoutePreview", () => {
  it("should render", () => {
    render(
      <RoutePreview pathValue="/api/v1" targetValue="https://target.com" includeSubpath={false} />
    );
    expect(screen.getByText("Route Preview")).toBeInTheDocument();
  });

  it("should return null when pathValue is empty", () => {
    const { container } = render(
      <RoutePreview pathValue="" targetValue="https://target.com" includeSubpath={false} />
    );
    expect(container.innerHTML).toBe("");
  });

  it("should return null when targetValue is empty", () => {
    const { container } = render(
      <RoutePreview pathValue="/api/v1" targetValue="" includeSubpath={false} />
    );
    expect(container.innerHTML).toBe("");
  });

  it("should display the full proxy URL for the endpoint", () => {
    render(
      <RoutePreview pathValue="/api/v1" targetValue="https://target.com" includeSubpath={false} />
    );
    expect(screen.getByText("https://proxy.example.com/api/v1")).toBeInTheDocument();
  });

  it("should display target URL in the forwards-to section", () => {
    render(
      <RoutePreview pathValue="/api/v1" targetValue="https://target.com" includeSubpath={false} />
    );
    expect(screen.getByText("https://target.com")).toBeInTheDocument();
  });

  it("should show subpath routing info when includeSubpath is true", () => {
    render(
      <RoutePreview pathValue="/api/v1" targetValue="https://target.com" includeSubpath={true} />
    );
    expect(screen.getByText("With subpaths:")).toBeInTheDocument();
  });

  it("should show the 'enable subpaths' hint when includeSubpath is false", () => {
    render(
      <RoutePreview pathValue="/api/v1" targetValue="https://target.com" includeSubpath={false} />
    );
    expect(screen.getByText(/not seeing the routing you wanted/i)).toBeInTheDocument();
  });
});
