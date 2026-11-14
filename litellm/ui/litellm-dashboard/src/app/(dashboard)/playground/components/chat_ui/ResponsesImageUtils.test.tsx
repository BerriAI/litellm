import { describe, expect, it } from "vitest";
import { createMultimodalMessage } from "./ResponsesImageUtils";

describe("createMultimodalMessage", () => {
  it("sends the uploaded image as a Responses input_image with detail, which strict servers require", async () => {
    const file = new File(["test content"], "test.png", { type: "image/png" });

    const result = await createMultimodalMessage("What is in this image?", file);

    expect(result.content).toEqual([
      { type: "input_text", text: "What is in this image?" },
      { type: "input_image", image_url: expect.stringMatching(/^data:image\/png;base64,/), detail: "auto" },
    ]);
  });
});
