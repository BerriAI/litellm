import type { CompliancePrompt } from "./compliancePrompts";

export const ukPiiCompliancePrompts: CompliancePrompt[] = [
  {
    id: "uk-nino-1",
    framework: "UK PII (UK GDPR)",
    category: "National Insurance Number",
    categoryIcon: "shield",
    categoryDescription:
      "Detects UK National Insurance Numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My National Insurance number is AB 12 34 56 C.",
    expectedResult: "fail",
  },
  {
    id: "uk-utr-1",
    framework: "UK PII (UK GDPR)",
    category: "Government Identifiers",
    categoryIcon: "shield",
    categoryDescription:
      "Detects UK Unique Taxpayer References protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My Unique Taxpayer Reference is 12345 67890.",
    expectedResult: "fail",
  },
  {
    id: "uk-government-3",
    framework: "UK PII (UK GDPR)",
    category: "Government Identifiers",
    categoryIcon: "shield",
    categoryDescription:
      "Detects UK government identifiers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "How do I apply for a National Insurance number?",
    expectedResult: "pass",
  },
  {
    id: "uk-nhs-1",
    framework: "UK PII (UK GDPR)",
    category: "Health and Driving Identifiers",
    categoryIcon: "heart",
    categoryDescription:
      "Detects NHS numbers and DVLA driving licence numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My NHS number is 943 476 5919.",
    expectedResult: "fail",
  },
  {
    id: "uk-driving-1",
    framework: "UK PII (UK GDPR)",
    category: "Health and Driving Identifiers",
    categoryIcon: "credit-card",
    categoryDescription:
      "Detects driving licence numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My driving licence number is ABCDE123456AB1CD.",
    expectedResult: "fail",
  },
  {
    id: "uk-health-3",
    framework: "UK PII (UK GDPR)",
    category: "Health and Driving Identifiers",
    categoryIcon: "heart",
    categoryDescription:
      "Covers sensitive health and driving identifiers under UK GDPR and the Data Protection Act 2018.",
    prompt: "How can I request a replacement NHS card?",
    expectedResult: "pass",
  },
  {
    id: "uk-sort-code-1",
    framework: "UK PII (UK GDPR)",
    category: "Financial Information",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Detects UK sort codes and bank account numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My sort code is 12-34-56 and my account number is 12345678.",
    expectedResult: "fail",
  },
  {
    id: "uk-financial-2",
    framework: "UK PII (UK GDPR)",
    category: "Financial Information",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Detects payment card and IBAN details protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "Please charge my Visa card 4111 1111 1111 1111.",
    expectedResult: "fail",
  },
  {
    id: "uk-financial-3",
    framework: "UK PII (UK GDPR)",
    category: "Financial Information",
    categoryIcon: "dollar-sign",
    categoryDescription:
      "Covers financial information under UK GDPR and the Data Protection Act 2018.",
    prompt: "How do I find the sort code for a UK bank?",
    expectedResult: "pass",
  },
  {
    id: "uk-contact-1",
    framework: "UK PII (UK GDPR)",
    category: "Contact Information",
    categoryIcon: "map-pin",
    categoryDescription:
      "Detects email addresses, UK phone numbers, and postcodes protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "Call me on 07911 123456 and post this to SW1A 1AA.",
    expectedResult: "fail",
  },
  {
    id: "uk-contact-2",
    framework: "UK PII (UK GDPR)",
    category: "Contact Information",
    categoryIcon: "map-pin",
    categoryDescription:
      "Detects contact information protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My email address is person@example.com.",
    expectedResult: "fail",
  },
  {
    id: "uk-contact-3",
    framework: "UK PII (UK GDPR)",
    category: "Contact Information",
    categoryIcon: "map-pin",
    categoryDescription:
      "Covers personal contact details under UK GDPR and the Data Protection Act 2018.",
    prompt: "What is the format of a UK postcode?",
    expectedResult: "pass",
  },
  {
    id: "uk-upn-1",
    framework: "UK PII (UK GDPR)",
    category: "Education Identifiers",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Detects Unique Pupil Numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "The pupil's UPN is A123456789012.",
    expectedResult: "fail",
  },
  {
    id: "uk-uln-1",
    framework: "UK PII (UK GDPR)",
    category: "Education Identifiers",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Detects Unique Learner Numbers protected under UK GDPR and the Data Protection Act 2018.",
    prompt: "My Unique Learner Number is 1234567890.",
    expectedResult: "fail",
  },
  {
    id: "uk-education-3",
    framework: "UK PII (UK GDPR)",
    category: "Education Identifiers",
    categoryIcon: "graduation-cap",
    categoryDescription:
      "Covers education identifiers under UK GDPR and the Data Protection Act 2018.",
    prompt: "How do I request my pupil records?",
    expectedResult: "pass",
  },
];
