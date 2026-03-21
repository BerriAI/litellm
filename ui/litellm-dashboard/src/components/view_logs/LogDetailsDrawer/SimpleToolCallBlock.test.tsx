import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SimpleToolCallBlock } from "./SimpleToolCallBlock";

describe("SimpleToolCallBlock", () => {
  it("should render the tool name", () => {
    render(
      <SimpleToolCallBlock
        tool={{ id: "1", name: "get_weather", arguments: {} }}
      />
    );
    expect(screen.getByText("get_weather")).toBeInTheDocument();
  });

  it('should display "function" badge', () => {
    render(
      <SimpleToolCallBlock
        tool={{ id: "1", name: "get_weather", arguments: {} }}
      />
    );
    expect(screen.getByText("function")).toBeInTheDocument();
  });

  it("should render arguments when present", () => {
    render(
      <SimpleToolCallBlock
        tool={{
          id: "1",
          name: "get_weather",
          arguments: { city: "London", units: "metric" },
        }}
      />
    );
    expect(screen.getByText("city:")).toBeInTheDocument();
    expect(screen.getByText('"London"')).toBeInTheDocument();
    expect(screen.getByText("units:")).toBeInTheDocument();
    expect(screen.getByText('"metric"')).toBeInTheDocument();
  });

  it("should not render arguments section when arguments are empty", () => {
    const { container } = render(
      <SimpleToolCallBlock
        tool={{ id: "1", name: "get_weather", arguments: {} }}
      />
    );
    // The tool name and "function" badge should be there, but no key: value pairs
    expect(screen.getByText("get_weather")).toBeInTheDocument();
    expect(screen.queryByText(/:$/)).not.toBeInTheDocument();
  });
});
