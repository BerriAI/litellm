import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import ChatUI from "./ChatUI";

describe("ChatUI", () => {
  it("should render the chat UI", () => {
    const { getByText } = render(
      <ChatUI
        accessToken="1234567890"
        token="1234567890"
        userRole="user"
        userID="1234567890"
        disabledPersonalKeyCreation={false}
      />,
    );
    expect(getByText("Test Key")).toBeInTheDocument();
  });
});
