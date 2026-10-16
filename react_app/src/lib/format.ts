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
