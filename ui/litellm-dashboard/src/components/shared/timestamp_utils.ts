import moment from "moment"

export const parseTimestamp = (timestamp: string): Date | null => {
  // Try different timestamp formats
  const formats = [
    "DD/MM/YYYY HH:mm:ss.SSS",
    "DD/MM/YYYY HH:mm:ss",
    "DD/MM/YYYY HH:mm",
    "DD/MM/YYYY",
    "YYYY-MM-DD HH:mm:ss.SSS",
    "YYYY-MM-DD HH:mm:ss",
    "YYYY-MM-DD HH:mm",
    "YYYY-MM-DD",
  ]

  for (const format of formats) {
    const parsed = moment(timestamp, format, true)
    if (parsed.isValid()) {
      return parsed.toDate()
    }
  }

  // Try ISO format
  const isoDate = moment(timestamp)
  if (isoDate.isValid()) {
    return isoDate.toDate()
  }

  return null
}

export const formatTimestamp = (date: Date, format: string = "DD/MM/YYYY HH:mm:ss.SSS"): string => {
  return moment(date).format(format)
}

export const validateTimestampInput = (input: string): { isValid: boolean; error?: string } => {
  const parsed = parseTimestamp(input)

  if (!parsed) {
    return {
      isValid: false,
      error: "Invalid timestamp format. Use DD/MM/YYYY HH:mm:ss.SSS or ISO format.",
    }
  }

  if (parsed > new Date()) {
    return {
      isValid: false,
      error: "Timestamp cannot be in the future.",
    }
  }

  return { isValid: true }
}
