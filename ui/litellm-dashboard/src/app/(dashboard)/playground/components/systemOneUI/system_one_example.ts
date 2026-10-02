import type { SystemOneRequest } from "./system_one_schemas";

export const SYSTEM_ONE_EXAMPLE: SystemOneRequest = {
  model: "jev-latest",
  state:
    "The headphones sound great and the battery easily lasts a full workday, but the left ear cushion started peeling after two weeks. I'd still buy them again if the build quality were better.",
  questions: {
    main_topic: {
      type: "choice",
      instructions: "What is this review mainly about?",
      criteria: {
        sound_quality: "How the product sounds",
        battery_life: "How long the battery lasts",
        build_quality: "Durability, materials, and defects",
        comfort: "How the product feels to wear",
      },
    },
    would_buy_again: {
      type: "noul",
      instructions: "Would the reviewer buy this product again?",
      criteria: {
        true: "They say they would buy it again or recommend it",
        false: "They say they would not buy it again or warn others away",
      },
    },
    overall_sentiment: {
      type: "score",
      instructions: "How positive is the review overall?",
      criteria: ["Very negative", "Mostly negative", "Mixed", "Mostly positive", "Very positive"],
    },
  },
};
