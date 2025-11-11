import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import CacheFieldGroup from "./CacheFieldGroup";

describe("CacheFieldGroup", () => {
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

  beforeEach(() => {
    cleanup();
  });

  it("should filter and render fields based on redisType", () => {
    /**
     * Tests that CacheFieldGroup filters fields based on redis_type and redisType prop.
     * This is the core functionality that shows/hides fields based on Redis deployment type.
     */
    const fields = [
      {
        field_name: "host",
        field_type: "String",
        ui_field_name: "Host",
        redis_type: null, // Applies to all types
      },
      {
        field_name: "redis_startup_nodes",
        field_type: "List",
        ui_field_name: "Startup Nodes",
        redis_type: "cluster", // Only for cluster
      },
      {
        field_name: "sentinel_nodes",
        field_type: "List",
        ui_field_name: "Sentinel Nodes",
        redis_type: "sentinel", // Only for sentinel
      },
    ];

    const cacheSettings = {
      host: "localhost",
      redis_startup_nodes: [],
    };

    // Test with cluster type - should show host and redis_startup_nodes
    const { rerender } = render(
      <CacheFieldGroup title="Cluster Settings" fields={fields} cacheSettings={cacheSettings} redisType="cluster" />,
    );

    expect(screen.getByText("Cluster Settings")).toBeInTheDocument();
    expect(screen.getAllByText("Host")).toHaveLength(1);
    expect(screen.getByText("Startup Nodes")).toBeInTheDocument();
    expect(screen.queryByText("Sentinel Nodes")).not.toBeInTheDocument();

    // Test with sentinel type - should show host and sentinel_nodes
    rerender(
      <CacheFieldGroup title="Sentinel Settings" fields={fields} cacheSettings={cacheSettings} redisType="sentinel" />,
    );

    expect(screen.getByText("Sentinel Settings")).toBeInTheDocument();
    expect(screen.getAllByText("Host")).toHaveLength(1);
    expect(screen.getByText("Sentinel Nodes")).toBeInTheDocument();
    expect(screen.queryByText("Startup Nodes")).not.toBeInTheDocument();

    // Test with node type - should only show host
    rerender(<CacheFieldGroup title="Node Settings" fields={fields} cacheSettings={cacheSettings} redisType="node" />);

    expect(screen.getByText("Node Settings")).toBeInTheDocument();
    expect(screen.getAllByText("Host")).toHaveLength(1);
    expect(screen.queryByText("Startup Nodes")).not.toBeInTheDocument();
    expect(screen.queryByText("Sentinel Nodes")).not.toBeInTheDocument();
  });

  it("should return null when no fields are visible", () => {
    /**
     * Tests that CacheFieldGroup returns null when no fields match the redisType.
     * This prevents rendering empty sections in the UI.
     */
    const fields = [
      {
        field_name: "redis_startup_nodes",
        field_type: "List",
        ui_field_name: "Startup Nodes",
        redis_type: "cluster", // Only for cluster
      },
    ];

    const cacheSettings = {};

    const { container } = render(
      <CacheFieldGroup
        title="Cluster Settings"
        fields={fields}
        cacheSettings={cacheSettings}
        redisType="node" // No fields match this type
      />,
    );

    // Component should return null, so container should be empty
    expect(container.firstChild).toBeNull();
  });

  it("should use field_default when currentValue is not available", () => {
    /**
     * Tests that CacheFieldGroup falls back to field_default when currentValue is missing.
     * This ensures fields display default values when cache settings are not set.
     */
    const fields = [
      {
        field_name: "port",
        field_type: "Integer",
        ui_field_name: "Port",
        field_default: 6379,
        redis_type: null,
      },
    ];

    const cacheSettings = {}; // No port value set

    render(
      <CacheFieldGroup title="Connection Settings" fields={fields} cacheSettings={cacheSettings} redisType="node" />,
    );

    const input = screen.getByRole("spinbutton", { name: "" });
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("name", "port");
    expect(input).toHaveValue(6379);
  });
});
