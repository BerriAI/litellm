import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import AudioRenderer from "./AudioRenderer";
import { MessageType } from "./types";

describe("AudioRenderer", () => {
  it("should render the audio renderer", () => {
    const { container } = render(
      <AudioRenderer message={{ content: "Hello, world!", role: "user", type: "audio" } as unknown as MessageType} />,
    );
    expect(container).toBeTruthy();
  });
});
