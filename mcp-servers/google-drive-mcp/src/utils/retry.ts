// Retry/timeout helpers. The implementation lives in ./retry.core.js (plain JS)
// so standalone Node scripts can reuse it without a build step; this module is the
// typed entry point the app imports. See ./retry.core.js for the policy details.
export { TimeoutError, withRetry } from './retry.core.js';
