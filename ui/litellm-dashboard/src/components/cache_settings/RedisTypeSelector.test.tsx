import { describe, expect, it } from "vitest";
import RedisTypeSelector from "./RedisTypeSelector";
import { render } from "@testing-library/react";

describe("RedisTypeSelector", () => {
  it("should render the component", () => {
    const { getAllByText } = render(
      <RedisTypeSelector redisType="redis" redisTypeDescriptions={{}} onTypeChange={() => {}} />,
    );
    expect(getAllByText(/Redis/i).length).toBeGreaterThan(0);
  });
});
