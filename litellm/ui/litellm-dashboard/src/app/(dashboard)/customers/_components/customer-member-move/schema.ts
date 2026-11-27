import { z } from "zod";
import { toMemberMoveBody } from "./mapper";

export const memberMoveSchema = z.object({ teams: z.array(z.string()).min(1) }).transform(toMemberMoveBody);
