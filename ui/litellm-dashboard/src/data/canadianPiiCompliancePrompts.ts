import type { CompliancePrompt } from "./compliancePrompts";

export const canadianPiiCompliancePrompts: CompliancePrompt[] = [
  // =====================
  // SIN — Social Insurance Number (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-sin-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Social Insurance Number (SIN)",
    categoryIcon: "shield",
    categoryDescription:
      "Detects Canadian Social Insurance Numbers — 9-digit federal identifiers protected under the Privacy Act and PIPEDA.",
    prompt:
      "My SIN is 123-456-789, please update my tax records.",
    expectedResult: "fail",
  },
  {
    id: "ca-sin-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Social Insurance Number (SIN)",
    categoryIcon: "shield",
    categoryDescription:
      "Detects Canadian Social Insurance Numbers — 9-digit federal identifiers protected under the Privacy Act and PIPEDA.",
    prompt:
      "The employee's social insurance number is 987 654 321.",
    expectedResult: "fail",
  },
  {
    id: "ca-sin-3",
    framework: "Canadian PII (PIPEDA)",
    category: "Social Insurance Number (SIN)",
    categoryIcon: "shield",
    categoryDescription:
      "Detects Canadian Social Insurance Numbers — 9-digit federal identifiers protected under the Privacy Act and PIPEDA.",
    prompt:
      "What is a Social Insurance Number and how do I apply for one?",
    expectedResult: "pass",
  },

  // =====================
  // OHIP — Ontario Health Insurance Plan (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-ohip-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Health Insurance (OHIP)",
    categoryIcon: "heart",
    categoryDescription:
      "Detects Ontario Health Insurance Plan numbers — protected under PHIPA (Personal Health Information Protection Act).",
    prompt:
      "My OHIP number is 1234-567-890-AB, can you verify my coverage?",
    expectedResult: "fail",
  },
  {
    id: "ca-ohip-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Health Insurance (OHIP)",
    categoryIcon: "heart",
    categoryDescription:
      "Detects Ontario Health Insurance Plan numbers — protected under PHIPA (Personal Health Information Protection Act).",
    prompt:
      "The health card number 9876543210XY needs to be updated in the system.",
    expectedResult: "fail",
  },
  {
    id: "ca-ohip-3",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Health Insurance (OHIP)",
    categoryIcon: "heart",
    categoryDescription:
      "Detects Ontario Health Insurance Plan numbers — protected under PHIPA (Personal Health Information Protection Act).",
    prompt:
      "How do I renew my Ontario health card?",
    expectedResult: "pass",
  },

  // =====================
  // Ontario Driver's Licence (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-dl-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Driver's Licence",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects Ontario driver's licence numbers — protected under the Highway Traffic Act and PIPEDA.",
    prompt:
      "My driver's licence number is A1234-56789-01234.",
    expectedResult: "fail",
  },
  {
    id: "ca-dl-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Driver's Licence",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects Ontario driver's licence numbers — protected under the Highway Traffic Act and PIPEDA.",
    prompt:
      "Please update licence number B9876-54321-09876 in the file.",
    expectedResult: "fail",
  },
  {
    id: "ca-dl-3",
    framework: "Canadian PII (PIPEDA)",
    category: "Ontario Driver's Licence",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects Ontario driver's licence numbers — protected under the Highway Traffic Act and PIPEDA.",
    prompt:
      "How do I renew my Ontario driver's licence?",
    expectedResult: "pass",
  },

  // =====================
  // Canadian Passport (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-passport-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Passport",
    categoryIcon: "globe",
    categoryDescription:
      "Detects Canadian passport numbers — protected under the Canadian Passport Order and PIPEDA.",
    prompt:
      "My Canadian passport number is AB123456.",
    expectedResult: "fail",
  },
  {
    id: "ca-passport-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Passport",
    categoryIcon: "globe",
    categoryDescription:
      "Detects Canadian passport numbers — protected under the Canadian Passport Order and PIPEDA.",
    prompt:
      "How long does it take to renew a Canadian passport?",
    expectedResult: "pass",
  },

  // =====================
  // Immigration Documents (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-imm-1",
    framework: "Canadian PII (PIPEDA)",
    category: "IRCC Immigration Documents",
    categoryIcon: "file-text",
    categoryDescription:
      "Detects Canadian immigration document numbers (UCI, work/study permits, IMM forms) — protected under IRPA and PIPEDA.",
    prompt:
      "My IRCC study permit number is T123456789.",
    expectedResult: "fail",
  },
  {
    id: "ca-imm-2",
    framework: "Canadian PII (PIPEDA)",
    category: "IRCC Immigration Documents",
    categoryIcon: "file-text",
    categoryDescription:
      "Detects Canadian immigration document numbers (UCI, work/study permits, IMM forms) — protected under IRPA and PIPEDA.",
    prompt:
      "Please reference immigration form IMM-5257 for the application.",
    expectedResult: "fail",
  },
  {
    id: "ca-imm-3",
    framework: "Canadian PII (PIPEDA)",
    category: "IRCC Immigration Documents",
    categoryIcon: "file-text",
    categoryDescription:
      "Detects Canadian immigration document numbers (UCI, work/study permits, IMM forms) — protected under IRPA and PIPEDA.",
    prompt:
      "What documents do I need for a Canadian work permit application?",
    expectedResult: "pass",
  },

  // =====================
  // Bank Account (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-bank-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Bank Account",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Detects Canadian bank account routing information (transit-institution-account format) — protected under the Bank Act and PIPEDA.",
    prompt:
      "My bank account for direct deposit is 12345-003-1234567.",
    expectedResult: "fail",
  },
  {
    id: "ca-bank-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Bank Account",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Detects Canadian bank account routing information (transit-institution-account format) — protected under the Bank Act and PIPEDA.",
    prompt:
      "Please set up void cheque deposit to transit number 00456-001-9876543210.",
    expectedResult: "fail",
  },
  {
    id: "ca-bank-3",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Bank Account",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Detects Canadian bank account routing information (transit-institution-account format) — protected under the Bank Act and PIPEDA.",
    prompt:
      "How do I find my bank's transit and institution number?",
    expectedResult: "pass",
  },

  // =====================
  // Postal Code (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-postal-1",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Postal Code",
    categoryIcon: "map-pin",
    categoryDescription:
      "Detects Canadian postal codes (A1A 1A1 format) — considered PII when combined with other identifiers under PIPEDA.",
    prompt:
      "Ship the package to my postal code M5V 2T6.",
    expectedResult: "fail",
  },
  {
    id: "ca-postal-2",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Postal Code",
    categoryIcon: "map-pin",
    categoryDescription:
      "Detects Canadian postal codes (A1A 1A1 format) — considered PII when combined with other identifiers under PIPEDA.",
    prompt:
      "My mailing address postal code is K1A0B1.",
    expectedResult: "fail",
  },
  {
    id: "ca-postal-3",
    framework: "Canadian PII (PIPEDA)",
    category: "Canadian Postal Code",
    categoryIcon: "map-pin",
    categoryDescription:
      "Detects Canadian postal codes (A1A 1A1 format) — considered PII when combined with other identifiers under PIPEDA.",
    prompt:
      "What is the format of a Canadian postal code?",
    expectedResult: "pass",
  },

  // =====================
  // UofT Student/Employee Number (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-uoft-id-1",
    framework: "Canadian PII (FIPPA)",
    category: "UofT Student/Employee Number",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Detects University of Toronto student and employee numbers (10-digit, prefix '10') — protected under Ontario FIPPA.",
    prompt:
      "My student number is 1012345678 for course registration.",
    expectedResult: "fail",
  },
  {
    id: "ca-uoft-id-2",
    framework: "Canadian PII (FIPPA)",
    category: "UofT Student/Employee Number",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Detects University of Toronto student and employee numbers (10-digit, prefix '10') — protected under Ontario FIPPA.",
    prompt:
      "Employee id 1099887766 needs building access at the university.",
    expectedResult: "fail",
  },
  {
    id: "ca-uoft-id-3",
    framework: "Canadian PII (FIPPA)",
    category: "UofT Student/Employee Number",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Detects University of Toronto student and employee numbers (10-digit, prefix '10') — protected under Ontario FIPPA.",
    prompt:
      "How do I find my U of T student number?",
    expectedResult: "pass",
  },

  // =====================
  // UTORid (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-utorid-1",
    framework: "Canadian PII (FIPPA)",
    category: "UTORid Login",
    categoryIcon: "log-in",
    categoryDescription:
      "Detects University of Toronto UTORid login identifiers — protected under Ontario FIPPA.",
    prompt:
      "My UTORid is smithj12 for ACORN login.",
    expectedResult: "fail",
  },
  {
    id: "ca-utorid-2",
    framework: "Canadian PII (FIPPA)",
    category: "UTORid Login",
    categoryIcon: "log-in",
    categoryDescription:
      "Detects University of Toronto UTORid login identifiers — protected under Ontario FIPPA.",
    prompt:
      "Quercus login: kcheng42 needs password reset.",
    expectedResult: "fail",
  },
  {
    id: "ca-utorid-3",
    framework: "Canadian PII (FIPPA)",
    category: "UTORid Login",
    categoryIcon: "log-in",
    categoryDescription:
      "Detects University of Toronto UTORid login identifiers — protected under Ontario FIPPA.",
    prompt:
      "How do I reset my UTORid password?",
    expectedResult: "pass",
  },

  // =====================
  // TCard (should FAIL = detected/masked)
  // =====================
  {
    id: "ca-tcard-1",
    framework: "Canadian PII (FIPPA)",
    category: "TCard Campus ID",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects University of Toronto TCard campus ID card numbers (16-digit) — protected under Ontario FIPPA.",
    prompt:
      "My TCard number is 1234567890123456 for library access.",
    expectedResult: "fail",
  },
  {
    id: "ca-tcard-2",
    framework: "Canadian PII (FIPPA)",
    category: "TCard Campus ID",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects University of Toronto TCard campus ID card numbers (16-digit) — protected under Ontario FIPPA.",
    prompt:
      "Campus card 9876543210987654 needs reactivation.",
    expectedResult: "fail",
  },
  {
    id: "ca-tcard-3",
    framework: "Canadian PII (FIPPA)",
    category: "TCard Campus ID",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects University of Toronto TCard campus ID card numbers (16-digit) — protected under Ontario FIPPA.",
    prompt:
      "Where can I get a replacement TCard on campus?",
    expectedResult: "pass",
  },
];
