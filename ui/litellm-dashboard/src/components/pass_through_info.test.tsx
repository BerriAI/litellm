import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PassThroughInfoView from "./pass_through_info";

vi.mock("./networking", () => ({
  getProxyBaseUrl: () => "https://proxy.example.com",
  updatePassThroughEndpoint: vi.fn(),
  deletePassThroughEndpointsCall: vi.fn(),
}));

vi.mock("./common_components/PassThroughSecuritySection", () => ({
  default: () => <div data-testid="security-section" />,
}));

vi.mock("./common_components/PassThroughGuardrailsSection", () => ({
  default: () => <div data-testid="guardrails-section" />,
}));

const mockEndpoint = {
  id: "ep-123",
  path: "/custom/api",
  target: "https://target.example.com",
  headers: { Authorization: "Bearer token123" },
  include_subpath: true,
  cost_per_request: 0.01,
  auth: true,
  methods: ["GET", "POST"],
};

describe("PassThroughInfoView", () => {
  const defaultProps = {
    endpointData: mockEndpoint,
    onClose: vi.fn(),
    accessToken: "test-token",
    isAdmin: true,
  };

  it("should render", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getByText(/pass through endpoint: \/custom\/api/i)).toBeInTheDocument();
  });

  it("should display the endpoint ID", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getByText("ep-123")).toBeInTheDocument();
  });

  it("should show Include Subpath badge when enabled", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getAllByText("Include Subpath").length).toBeGreaterThanOrEqual(1);
  });

  it("should show Auth Required badge when auth is true", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getByText("Auth Required")).toBeInTheDocument();
  });

  it("should display HTTP methods as badges", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getByText("GET")).toBeInTheDocument();
    expect(screen.getByText("POST")).toBeInTheDocument();
  });

  it("should show 'All HTTP methods supported' when no methods specified", () => {
    const noMethodEndpoint = { ...mockEndpoint, methods: [] };
    render(<PassThroughInfoView {...defaultProps} endpointData={noMethodEndpoint} />);
    expect(screen.getByText("All HTTP methods supported")).toBeInTheDocument();
  });

  it("should show headers count badge", () => {
    render(<PassThroughInfoView {...defaultProps} />);
    expect(screen.getByText("1 headers configured")).toBeInTheDocument();
  });

  it("should call onClose when Back button is clicked", async () => {
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(<PassThroughInfoView {...defaultProps} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /back/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("should show Settings tab for admin users", () => {
    render(<PassThroughInfoView {...defaultProps} isAdmin={true} />);
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("should not show Settings tab for non-admin users", () => {
    render(<PassThroughInfoView {...defaultProps} isAdmin={false} />);
    expect(screen.queryByText("Settings")).not.toBeInTheDocument();
  });
});
