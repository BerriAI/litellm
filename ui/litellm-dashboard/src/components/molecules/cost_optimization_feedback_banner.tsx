import React, { useState } from "react";
import { CommentOutlined } from "@ant-design/icons";

const STORAGE_KEY = "hideCostOptimizationFeedbackBanner";
const DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32172";

const CostOptimizationFeedbackBanner: React.FC = () => {
  const [dismissed, setDismissed] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(STORAGE_KEY) === "true";
    }
    return false;
  });

  if (dismissed) {
    return null;
  }

  return (
    <div className="mb-4 px-4 py-3 bg-blue-50 rounded-lg border border-blue-100 flex items-center gap-4">
      <div className="shrink-0 w-10 h-10 bg-white rounded-full flex items-center justify-center border border-blue-200">
        <CommentOutlined style={{ fontSize: "18px", color: "#6366f1" }} />
      </div>
      <div className="flex-1 min-w-0">
        <h4 className="text-gray-900 font-semibold text-sm m-0">Help shape cost optimization</h4>
        <p className="text-gray-500 text-xs m-0 mt-0.5">
          We&apos;re collecting suggestions for cost optimization improvements across routing, budgets, and more. Let us
          know what you&apos;d like to see.
        </p>
      </div>
      <a
        href={DISCUSSION_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="shrink-0 inline-flex items-center gap-2 px-4 py-2 bg-[#6366f1] hover:bg-[#5558e3] text-white text-sm font-medium rounded-lg transition-colors"
      >
        Share Feedback
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
          />
        </svg>
      </a>
      <button
        onClick={() => {
          setDismissed(true);
          localStorage.setItem(STORAGE_KEY, "true");
        }}
        className="shrink-0 p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full transition-colors"
        aria-label="Dismiss banner"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-5 w-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
};

export default CostOptimizationFeedbackBanner;
