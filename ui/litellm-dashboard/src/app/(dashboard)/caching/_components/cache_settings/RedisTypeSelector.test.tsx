import { describe, expect, it } from "vitest";
import RedisTypeSelector from "./RedisTypeSelector";
import { render, screen } from "@testing-library/react";

describe("RedisTypeSelector", () => {
  it("should render the component", () => {
    render(<RedisTypeSelector redisType="redis" redisTypeDescriptions={{}} onTypeChange={() => {}} />);
    expect(screen.getAllByText(/Redis/i).length).toBeGreaterThan(0);
  });
});
