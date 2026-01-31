import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import CacheFieldRenderer from "./CacheFieldRenderer";

// Mock the useAuthorized hook to avoid Next.js router dependency
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "mock-token",
    accessToken: "mock-access-token",
    userId: "mock-user-id",
    userEmail: "test@example.com",
    userRole: "Admin",
    premiumUser: false,
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  }),
}));

describe("CacheFieldRenderer", () => {
  it("should render a checkbox for Boolean field type", () => {
    /**
     * Tests that Boolean fields render as checkboxes with proper defaultChecked value.
     * This is the core functionality for boolean cache settings.
     */
    const field = {
      field_name: "ssl",
      field_type: "Boolean",
      ui_field_name: "Enable SSL",
      field_description: "Enable SSL encryption",
    };

    render(<CacheFieldRenderer field={field} currentValue={true} />);

    const checkbox = screen.getByRole("checkbox", { name: "" });
    expect(checkbox).toBeInTheDocument();
    expect(checkbox).toBeChecked();
    expect(screen.getByText("Enable SSL")).toBeInTheDocument();
    expect(screen.getByText("Enable SSL encryption")).toBeInTheDocument();
  });

  it("should render a textarea for List field type", () => {
    /**
     * Tests that List fields render as textareas with JSON stringified values.
     * This handles array/list cache settings like redis_startup_nodes.
     */
    const field = {
      field_name: "redis_startup_nodes",
      field_type: "List",
      ui_field_name: "Redis Startup Nodes",
      field_description: "List of Redis cluster nodes",
    };

    const currentValue = [
      { host: "localhost", port: 6379 },
      { host: "localhost", port: 6380 },
    ];

    render(<CacheFieldRenderer field={field} currentValue={currentValue} />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeInTheDocument();
    expect(textarea.tagName).toBe("TEXTAREA");
    expect(textarea).toHaveValue(JSON.stringify(currentValue, null, 2));
    expect(screen.getByText("Redis Startup Nodes")).toBeInTheDocument();
  });

  it("should render a password input for password field", () => {
    /**
     * Tests that password fields render as password inputs.
     * This ensures sensitive data is masked in the UI.
     */
    const field = {
      field_name: "password",
      field_type: "String",
      ui_field_name: "Password",
      field_description: "Redis password",
    };

    render(<CacheFieldRenderer field={field} currentValue="secret123" />);

    const input = screen.getByPlaceholderText("Redis password");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "password");
    expect(input).toHaveValue("secret123");
  });

  it("should render a number input for Integer field type", () => {
    /**
     * Tests that Integer fields render as number inputs.
     * This ensures proper validation for numeric cache settings.
     */
    const field = {
      field_name: "port",
      field_type: "Integer",
      ui_field_name: "Port",
      field_description: "Redis port number",
    };

    render(<CacheFieldRenderer field={field} currentValue={6379} />);

    const input = screen.getByPlaceholderText("Redis port number");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "number");
    expect(input).toHaveValue(6379);
  });

  it("should render a text input for String field type", () => {
    /**
     * Tests that String fields render as text inputs.
     * This is the default rendering for text-based cache settings.
     */
    const field = {
      field_name: "host",
      field_type: "String",
      ui_field_name: "Host",
      field_description: "Redis host address",
    };

    render(<CacheFieldRenderer field={field} currentValue="localhost" />);

    const input = screen.getByPlaceholderText("Redis host address");
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("type", "text");
    expect(input).toHaveValue("localhost");
  });
});
