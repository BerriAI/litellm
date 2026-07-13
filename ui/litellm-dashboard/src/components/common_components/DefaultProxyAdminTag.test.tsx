import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import DefaultProxyAdminTag from "./DefaultProxyAdminTag";

describe("DefaultProxyAdminTag", () => {
  it("should render", () => {
    render(<DefaultProxyAdminTag userId="some-user" />);
    expect(screen.getByText("some-user")).toBeInTheDocument();
  });

  it("should render a blue tag when userId is default_user_id", () => {
    render(<DefaultProxyAdminTag userId="default_user_id" />);
    expect(screen.getByText("Default Proxy Admin")).toBeInTheDocument();
  });

  it("should render plain text when userId is a regular value", () => {
    render(<DefaultProxyAdminTag userId="alice@example.com" />);
    expect(screen.getByText("alice@example.com")).toBeInTheDocument();
    expect(screen.queryByText("Default Proxy Admin")).not.toBeInTheDocument();
  });

  it("should render empty text when userId is null", () => {
    const { container } = render(<DefaultProxyAdminTag userId={null} />);
    expect(screen.queryByText("Default Proxy Admin")).not.toBeInTheDocument();
    expect(container.textContent).toBe("");
  });
});
