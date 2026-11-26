import { type RejectionValues } from "./schema";

export const toReviewNotes = ({ reviewNotes }: RejectionValues): string | undefined => reviewNotes || undefined;
