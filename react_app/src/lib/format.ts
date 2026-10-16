/** Port of `_format_timestamp`: unix seconds/ms -> "dd/mm/yyyy HH:MM:SS UTC". */
export function formatTimestamp(value: unknown): string {
  if (value === null || value === undefined || value === "" || value === "N/A") {
    return "N/A";
  }

  let ts = Number(value);
  if (Number.isNaN(ts)) return String(value);
  if (ts > 1e12) ts /= 1000;

  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return String(value);

  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${p(d.getUTCDate())}/${p(d.getUTCMonth() + 1)}/${d.getUTCFullYear()} ` +
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} UTC`
  );
}

/** Convert a YYYY-MM-DD date to a Unix timestamp in seconds (UTC). */
export function dateToUnixSeconds(date: string): number {
  const [year, month, day] = date.split("-").map(Number);

  return Date.UTC(year, month - 1, day) / 1000;
}

/** Get the exclusive UTC timestamp for the day after the selected end date. */
export function dateAfterToUnixSeconds(date: string): number {
  return dateToUnixSeconds(date) + 24 * 60 * 60;
}