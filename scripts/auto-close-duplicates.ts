#!/usr/bin/env bun

declare global {
  var process: {
    env: Record<string, string | undefined>;
  };
}

interface GitHubIssue {
  number: number;
  title: string;
  user: { login: string };
  labels: { name: string }[];
}

interface GitHubComment {
  id: number;
  body: string;
  created_at: string;
  user: { type: string };
}

interface GitHubReaction {
  user: { login: string };
  content: string;
}

const FLAG_LABEL = "potential-duplicate";
const FLAG_MARKER = "<!-- litellm:potential-duplicate";
const CANDIDATES = /<!-- litellm:potential-duplicate candidates=([\d,]*) -->/;
const GRACE_PERIOD_DAYS = 3;

async function githubRequest<T>(
  endpoint: string,
  token: string,
  method: string = "GET",
  body?: any,
): Promise<T> {
  const response = await fetch(`https://api.github.com${endpoint}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: "application/vnd.github.v3+json",
      "User-Agent": "auto-close-duplicates-script",
      ...(body && { "Content-Type": "application/json" }),
    },
    ...(body && { body: JSON.stringify(body) }),
  });

  if (!response.ok) {
    throw new Error(
      `GitHub API request failed: ${response.status} ${response.statusText}`,
    );
  }

  return response.json();
}

function extractDuplicateIssueNumber(
  commentBody: string,
  issueNumber: number,
): number | null {
  // Read candidates from the marker's digits-only field, never from the prose.
  // Titles are user-controlled and are interpolated into this same comment, so
  // scanning the body would let an issue titled "... see #1" redirect a closure
  // onto an unrelated report.
  const field = commentBody.match(CANDIDATES);
  if (!field) {
    return null;
  }

  const candidates = field[1]
    .split(",")
    .filter((value) => value !== "")
    .map(Number)
    .filter((value) => value !== issueNumber);

  // The detector orders candidates by score, not age, so the first one listed can
  // be newer than the original report. Duplicates fold into the earliest issue.
  return candidates.length > 0 ? Math.min(...candidates) : null;
}

async function closeIssueAsDuplicate(
  owner: string,
  repo: string,
  issueNumber: number,
  duplicateOfNumber: number,
  token: string,
): Promise<void> {
  await githubRequest(
    `/repos/${owner}/${repo}/issues/${issueNumber}/comments`,
    token,
    "POST",
    {
      body: `This issue has been automatically closed as a duplicate of #${duplicateOfNumber}.

The duplicate notice went unanswered for ${GRACE_PERIOD_DAYS} days. If this is incorrect, please re-open this issue and say how it differs from #${duplicateOfNumber}.

<!-- litellm:closed-as-duplicate -->`,
    },
  );

  // Added on its own endpoint rather than in the PATCH below, because sending
  // `labels` with the state change replaces every label on the issue.
  await githubRequest(
    `/repos/${owner}/${repo}/issues/${issueNumber}/labels`,
    token,
    "POST",
    { labels: ["duplicate"] },
  );

  await githubRequest(
    `/repos/${owner}/${repo}/issues/${issueNumber}`,
    token,
    "PATCH",
    { state: "closed", state_reason: "duplicate" },
  );
}

async function autoCloseDuplicates(): Promise<void> {
  console.log("[DEBUG] Starting auto-close duplicates script");

  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    throw new Error("GITHUB_TOKEN environment variable is required");
  }
  console.log("[DEBUG] GitHub token found");

  const owner = process.env.GITHUB_REPOSITORY_OWNER || "BerriAI";
  const repo = process.env.GITHUB_REPOSITORY_NAME || "litellm";
  console.log(`[DEBUG] Repository: ${owner}/${repo}`);

  const threeDaysAgo = new Date();
  threeDaysAgo.setDate(threeDaysAgo.getDate() - GRACE_PERIOD_DAYS);
  console.log(
    `[DEBUG] Checking for duplicate comments older than: ${threeDaysAgo.toISOString()}`,
  );

  // Only issues the detector labelled. Walking the whole open backlog would cost a
  // comments request per issue, which on a four-figure backlog exhausts the Actions
  // token's hourly rate limit for a handful of matches.
  console.log(`[DEBUG] Fetching open issues labelled '${FLAG_LABEL}'...`);
  const allIssues: GitHubIssue[] = [];
  let page = 1;
  const perPage = 100;

  while (true) {
    const pageIssues: GitHubIssue[] = await githubRequest(
      `/repos/${owner}/${repo}/issues?state=open&labels=${FLAG_LABEL}&per_page=${perPage}&page=${page}`,
      token,
    );

    if (pageIssues.length === 0) break;

    allIssues.push(...pageIssues);
    page++;

    // Safety limit to avoid infinite loops
    if (page > 20) break;
  }

  const issues = allIssues;
  console.log(`[DEBUG] Found ${issues.length} flagged issues`);

  let processedCount = 0;
  let candidateCount = 0;

  for (const issue of issues) {
    processedCount++;
    console.log(
      `[DEBUG] Processing issue #${issue.number} (${processedCount}/${issues.length}): ${issue.title}`,
    );

    console.log(`[DEBUG] Fetching comments for issue #${issue.number}...`);
    const comments: GitHubComment[] = await githubRequest(
      `/repos/${owner}/${repo}/issues/${issue.number}/comments?per_page=100`,
      token,
    );
    console.log(
      `[DEBUG] Issue #${issue.number} has ${comments.length} comments`,
    );

    // The author filter matters: GitHub's "Quote reply" carries the HTML marker into
    // a human comment, and treating that as a fresh notice restarts the clock.
    const dupeComments = comments.filter(
      (comment) =>
        comment.body.includes(FLAG_MARKER) && comment.user.type === "Bot",
    );
    console.log(
      `[DEBUG] Issue #${issue.number} has ${dupeComments.length} duplicate detection comments`,
    );

    if (dupeComments.length === 0) {
      console.log(
        `[DEBUG] Issue #${issue.number} - no duplicate comments found, skipping`,
      );
      continue;
    }

    const lastDupeComment = dupeComments[dupeComments.length - 1];
    const dupeCommentDate = new Date(lastDupeComment.created_at);
    console.log(
      `[DEBUG] Issue #${issue.number} - most recent duplicate comment from: ${dupeCommentDate.toISOString()}`,
    );

    if (dupeCommentDate > threeDaysAgo) {
      console.log(
        `[DEBUG] Issue #${issue.number} - duplicate comment is too recent, skipping`,
      );
      continue;
    }
    console.log(
      `[DEBUG] Issue #${issue.number} - duplicate comment is old enough (${Math.floor(
        (Date.now() - dupeCommentDate.getTime()) / (1000 * 60 * 60 * 24),
      )} days)`,
    );

    const commentsAfterDupe = comments.filter(
      (comment) => new Date(comment.created_at) > dupeCommentDate,
    );
    console.log(
      `[DEBUG] Issue #${issue.number} - ${commentsAfterDupe.length} comments after duplicate detection`,
    );

    if (commentsAfterDupe.length > 0) {
      console.log(
        `[DEBUG] Issue #${issue.number} - has activity after duplicate comment, skipping`,
      );
      continue;
    }

    console.log(
      `[DEBUG] Issue #${issue.number} - checking reactions on duplicate comment...`,
    );
    const reactions: GitHubReaction[] = await githubRequest(
      `/repos/${owner}/${repo}/issues/comments/${lastDupeComment.id}/reactions?per_page=100`,
      token,
    );
    console.log(
      `[DEBUG] Issue #${issue.number} - duplicate comment has ${reactions.length} reactions`,
    );

    const authorThumbsDown = reactions.some(
      (reaction) =>
        reaction.user.login === issue.user.login && reaction.content === "-1",
    );
    console.log(
      `[DEBUG] Issue #${issue.number} - author thumbs down reaction: ${authorThumbsDown}`,
    );

    if (authorThumbsDown) {
      console.log(
        `[DEBUG] Issue #${issue.number} - author disagreed with duplicate detection, skipping`,
      );
      continue;
    }

    const duplicateIssueNumber = extractDuplicateIssueNumber(
      lastDupeComment.body,
      issue.number,
    );
    if (!duplicateIssueNumber) {
      console.log(
        `[DEBUG] Issue #${issue.number} - could not extract duplicate issue number from comment, skipping`,
      );
      continue;
    }

    if (duplicateIssueNumber > issue.number) {
      console.log(
        `[DEBUG] Issue #${issue.number} - only candidate #${duplicateIssueNumber} is newer, skipping`,
      );
      continue;
    }

    candidateCount++;
    const issueUrl = `https://github.com/${owner}/${repo}/issues/${issue.number}`;

    try {
      console.log(
        `[INFO] Auto-closing issue #${issue.number} as duplicate of #${duplicateIssueNumber}: ${issueUrl}`,
      );
      await closeIssueAsDuplicate(
        owner,
        repo,
        issue.number,
        duplicateIssueNumber,
        token,
      );
      console.log(
        `[SUCCESS] Successfully closed issue #${issue.number} as duplicate of #${duplicateIssueNumber}`,
      );
    } catch (error) {
      console.error(
        `[ERROR] Failed to close issue #${issue.number} as duplicate: ${error}`,
      );
    }
  }

  console.log(
    `[DEBUG] Script completed. Processed ${processedCount} issues, found ${candidateCount} candidates for auto-close`,
  );
}

autoCloseDuplicates().catch(console.error);

// Make it a module
export {};
