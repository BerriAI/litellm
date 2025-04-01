import dotenv from "dotenv";

export default async function globalSetup() {
  dotenv.config({
    //path should be relative to playwright.config.ts
    path: "./../../.env",
  });
}
