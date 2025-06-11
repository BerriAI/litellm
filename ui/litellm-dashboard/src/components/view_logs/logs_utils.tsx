import moment from "moment";

// Add this function to format the time range display
export const getTimeRangeDisplay = (isCustomDate: boolean, startTime: string, endTime: string) => {
  if (isCustomDate) {
    return `${moment(startTime).format('MMM D, h:mm A')} - ${moment(endTime).format('MMM D, h:mm A')}`;
  }
  
  const now = moment();
  const start = moment(startTime);
  const diffMinutes = now.diff(start, 'minutes');
  
  if (diffMinutes <= 15) return 'Last 15 Minutes';
  if (diffMinutes <= 60) return 'Last Hour';
  
  const diffHours = now.diff(start, 'hours');
  if (diffHours <= 4) return 'Last 4 Hours';
  if (diffHours <= 24) return 'Last 24 Hours';
  if (diffHours <= 168) return 'Last 7 Days';
  return `${start.format('MMM D')} - ${now.format('MMM D')}`;
};