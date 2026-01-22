import { Role } from "./roles";

const isCI = !!process.env.CI;

export const users = {
  [Role.ProxyAdmin]: {
    email: "admin",
    password: isCI ? "gm" : "sk-1234",
  },
  [Role.InternalUserViewer]: {
    email: "internalViewer@test.com",
    password: "test",
  },
};
