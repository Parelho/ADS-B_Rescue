
import { useState } from "react";

interface DateRangePickerProps {
  onApply: (startDate: string, endDate: string) => void;
}

export default function DateRangePicker({
  onApply,
}: DateRangePickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const invalidRange =
    startDate !== "" &&
    endDate !== "" &&
    endDate < startDate;

  function handleApply() {
    if (!startDate || !endDate || invalidRange) return;

    onApply(startDate, endDate);
    setIsOpen(false);
  }

  return (
    <div className="relative inline-block">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="rounded-md border px-4 py-2 shadow-sm"
      >
        Select date range
      </button>

      {isOpen && (
        <div className="absolute left-0 top-full z-[1100] mt-2 w-72 rounded-lg border bg-white p-4 text-gray-900 shadow-lg">
          <h3 className="mb-3 font-semibold">
            Filter trajectory dates
          </h3>

          <label className="mb-1 block text-sm">
            Start date
          </label>
          <input
            type="date"
            value={startDate}
            max={endDate || undefined}
            onChange={(e) => setStartDate(e.target.value)}
            className="mb-4 w-full rounded-md border p-2"
          />

          <label className="mb-1 block text-sm">
            End date
          </label>
          <input
            type="date"
            value={endDate}
            min={startDate || undefined}
            onChange={(e) => setEndDate(e.target.value)}
            className="mb-3 w-full rounded-md border p-2"
          />

          {invalidRange && (
            <p className="mb-3 text-sm text-red-600">
              End date must be on or after start date.
            </p>
          )}

          <button
            type="button"
            onClick={handleApply}
            disabled={!startDate || !endDate || invalidRange}
            className="w-full rounded-md bg-blue-600 px-4 py-2 text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            Apply filter
          </button>
        </div>
      )}
    </div>
  );
}
