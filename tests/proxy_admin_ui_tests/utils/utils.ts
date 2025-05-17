// import * as dotenv from 'dotenv';
// import { config } from "dotenv";
// import path from "path";
// config({ path: "./../../../../.env.example" });

export function loginDetailsSet(): Boolean {
  //   console.log(process.env.DATABASE_URL);
  //   console.log(process.env.UI_PASSWORD);
  let loginDetailsSet = false;
  if (process.env.UI_USERNAME && process.env.UI_PASSWORD) {
    loginDetailsSet = true;
  }
  return loginDetailsSet;
}

export function generateRandomString(length) {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  let result = "";

  const randomSequence = new Uint8Array(length);
  crypto.getRandomValues(randomSequence);
  randomSequence.forEach((number) => {
    result += chars[number % chars.length];
  });

  return result;
}
