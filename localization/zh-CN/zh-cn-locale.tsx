"use client";

import type { ReactNode } from "react";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import moment from "moment";
import "moment/locale/zh-cn";
import { setDefaultOptions } from "date-fns";
import { zhCN } from "date-fns/locale";

dayjs.locale("zh-cn");
moment.locale("zh-cn");
setDefaultOptions({ locale: zhCN });

export default function ZhCnLocale({ children }: Readonly<{ children: ReactNode }>) {
  return <>{children}</>;
}
