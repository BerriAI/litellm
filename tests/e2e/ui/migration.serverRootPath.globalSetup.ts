import globalSetup from "./globalSetup";

export default async function migrationServerRootPathGlobalSetup() {
  if (!process.env.SERVER_ROOT_PATH) {
    throw new Error(
      "migration.serverRootPath.config.ts requires SERVER_ROOT_PATH to be set (e.g. SERVER_ROOT_PATH=/litellm). " +
        "Without it this config silently re-runs the default mount and never exercises the prefix. " +
        "For the root-less run use the default playwright.config.ts (npm run e2e:migration).",
    );
  }
  await globalSetup();
}
