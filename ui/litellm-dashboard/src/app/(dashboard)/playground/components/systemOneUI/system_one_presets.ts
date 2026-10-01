import type { SystemOneRequest } from "./system_one_types";

export interface SystemOnePreset {
  id: string;
  name: string;
  payload: SystemOneRequest;
}

export const SYSTEM_ONE_PRESETS: SystemOnePreset[] = [
  {
    id: "noul",
    name: "Noul: Support escalation",
    payload: {
      model: "jev-latest",
      state:
        "I've tried the password reset twice and still can't sign in. Please connect me with someone who can help.",
      questions: {
        needs_human: {
          type: "noul",
          instructions: "Is the person asking for direct help from a support representative?",
          criteria: {
            true: "They explicitly ask to speak with a person or representative",
            false: "They do not request a person to assist them",
          },
        },
      },
    },
  },
  {
    id: "choice",
    name: "Choice: Support routing",
    payload: {
      model: "jev-latest",
      state: "The invoice shows two charges for my monthly plan, but I only renewed once.",
      questions: {
        department: {
          type: "choice",
          instructions: "Which support team should handle this message?",
          criteria: {
            billing: "Invoices, payments, subscriptions, and refunds",
            account: "Login, profile, and account access",
            technical: "Product bugs, outages, and integrations",
          },
        },
      },
    },
  },
  {
    id: "score",
    name: "Score: Customer urgency",
    payload: {
      model: "jev-latest",
      state: "Our team is blocked from exporting reports before tomorrow's audit.",
      questions: {
        urgency: {
          type: "score",
          instructions: "Rate how urgently the support team should respond.",
          criteria: [
            "Routine request with no meaningful time pressure",
            "Some urgency, but a short delay is manageable",
            "Time-sensitive issue that is blocking important work",
            "Immediate response is needed to prevent significant impact",
          ],
        },
      },
    },
  },
  {
    id: "combined",
    name: "Combined: Support triage",
    payload: {
      model: "jev-latest",
      state: {
        message:
          "I was billed after cancelling, and I need the refund processed before my account renews again tomorrow.",
        account_tier: "standard",
      },
      questions: {
        department: {
          type: "choice",
          instructions: "Which support team should own the request?",
          criteria: {
            billing: "Charges, refunds, invoices, and subscription renewals",
            account: "Access, profile settings, and account ownership",
            technical: "Product behavior, errors, and integrations",
          },
        },
        needs_same_day_reply: {
          type: "noul",
          instructions: "Does the message indicate a response is needed today?",
          criteria: {
            true: "The customer identifies an upcoming renewal or deadline today",
            false: "There is no immediate deadline mentioned",
          },
        },
        urgency: {
          type: "score",
          instructions: "Rate the urgency of resolving the request.",
          criteria: [
            "No time pressure is described",
            "The customer would prefer a quick response",
            "A near-term deadline makes a prompt response important",
          ],
        },
      },
    },
  },
];
