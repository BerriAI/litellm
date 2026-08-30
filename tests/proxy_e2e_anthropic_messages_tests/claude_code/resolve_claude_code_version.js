const { execFileSync } = require("node:child_process");

const packageName = "@anthropic-ai/claude-code";
const minAgeMs = 3 * 24 * 60 * 60 * 1000;
const cutoff = Date.now() - minAgeMs;
const raw = execFileSync("npm", ["view", packageName, "time", "--json"], {
  encoding: "utf8",
});
const time = JSON.parse(raw);
const versions = Object.keys(time)
  .filter((version) => /^\d/.test(version))
  .filter((version) => new Date(time[version]).getTime() <= cutoff)
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

if (versions.length === 0) {
  console.error(
    `No ${packageName} version is older than the 3-day exclusion window.`
  );
  process.exit(1);
}

console.log(versions[versions.length - 1]);
