# Current-main integration proof

Captured October 27, 2026 against the repaired team-access worktree integrating main 34c10829b1f2aba5ee44887a413bac12d77e2bc0. Source hashes identify the files exercised. No unit-test doubles replaced the UI, gateway authorization, or database in this run

The browser used http://127.0.0.1:3030/ui/chat with the actual gateway on port 4234. An isolated local PostgreSQL database held a synthetic internal user and Research and Clinical teams, each granting team-chat-proof. The configured provider was real OpenAI gpt-5.5. The user had no direct grant for that model

1. Logged in through the gateway login endpoint, selected team-chat-proof, and uploaded document.txt. Send stayed disabled until selecting Research
2. Sent the document and asked for its synthetic project code. The answer was MARIGOLD-72, with the Research team header and a native input_file part
3. Asked for the code again under Research. The request reused previous_response_id and did not replay the file
4. Selected Clinical and asked to read the earlier document again. The request omitted previous_response_id and replayed the original document in the conversation. The answer was MARIGOLD-72
5. Switched to a 390px mobile viewport. The Clinical team selector remained visible and the document had no horizontal overflow
6. Returned to desktop, edited the original attached-document message, and clicked Save & Send under Clinical. The document remained attached and the answer was MARIGOLD-72

All four Responses requests returned HTTP 200. requests.json records only HTTP status, model, team, previous-response presence, and file metadata. billing.json records persisted user/team/request charges from Prisma after the real requests. Screenshots were visually inspected and show only synthetic content with no credentials. Browser resource origins showed the pre-existing Fontshare stylesheet and local application traffic, with no external Chat analytics collection observed

This is after-repair integration proof, not a new before/after reproduction of the original access bug. Branding was preserved by source comparison; no configured-logo runtime scenario was rerun. No GitHub publication, Slack message, telemetry event, or new collection feature was performed
