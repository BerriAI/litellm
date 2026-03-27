import React, { useState } from "react";
import styles from "./styles.module.css";

interface VersionEntry {
  version: string;
  sha256: string;
  gitCommit: string;
}

interface Props {
  entries: VersionEntry[];
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <button
      className={styles.copyBtn}
      onClick={handleCopy}
      title="Copy full SHA-256"
    >
      {copied ? "✓" : "⧉"}
    </button>
  );
}

export default function VersionVerificationTable({ entries }: Props) {
  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Version</th>
            <th>SHA-256</th>
            <th>Clean of IOCs</th>
            <th>Matches Git</th>
            <th>Git Commit</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={`${entry.version}-${entry.gitCommit}`}>
              <td className={styles.version}>{entry.version}</td>
              <td>
                <span className={styles.sha}>
                  <code>{entry.sha256.slice(0, 16)}…</code>
                  <CopyButton text={entry.sha256} />
                </span>
              </td>
              <td>
                <span className={styles.badgeClean}>✔ CLEAN</span>
              </td>
              <td>
                <span className={styles.badgeYes}>✔ YES</span>
              </td>
              <td>
                <a
                  className={styles.commitLink}
                  href={`https://github.com/BerriAI/litellm/commit/${entry.gitCommit}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  {entry.gitCommit}
                </a>
              </td>
              <td>
                <span className={styles.badgeClean}>✔ CLEAN</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
