import type { ReactNode } from "react";
import type { Airport, Selection } from "../types";
import { findNearbyAirport } from "../lib/airports";
import { formatTimestamp } from "../lib/format";

interface Props {
  selection: Selection | null;
  airports: Airport[];
  refreshing: boolean;
}

const panelClassName =
  "w-80 shrink-0 overflow-y-auto rounded-[10px] border border-gray-300 bg-[#f4f6f8] p-3 text-base max-[800px]:max-h-[35%] max-[800px]:w-full";

function DetailRow({
  label,
  value,
  icon,
}: {
  label: string;
  value: string | number;
  icon: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-center gap-3 rounded-lg border border-gray-200 bg-white p-3">
      <span
        className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-700"
        aria-hidden="true"
      >
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          {label}
        </div>
        <div className="break-words text-[15px] font-medium leading-snug text-gray-900">
          {value}
        </div>
      </div>
    </div>
  );
}

export default function InfoPanel({ selection, airports, refreshing }: Props) {
  if (refreshing) {
    return <aside className={panelClassName}>Atualizando dados da API...</aside>;
  }
  if (!selection) {
    return (
      <aside className={panelClassName}>
        Clique em um marcador para ver os detalhes do voo.
      </aside>
    );
  }

  const isPlane = selection.kind === "plane";
  const title = isPlane ? selection.plane.icao : selection.airport.name;
  const [lat, lon] = isPlane
    ? selection.plane.routePoints[selection.plane.routePoints.length - 1]
    : [selection.airport.lat, selection.airport.lon];

  // Same as the Python app: nearest airport to the *clicked marker position*.
  const nearby = findNearbyAirport(airports, lat, lon);

  return (
    <aside className={panelClassName}>
      <div className="mb-2 text-[19px] font-bold">{title}</div>
      <hr className="mb-3 border-t border-gray-300" />
      <div className="space-y-2">
        <DetailRow
          label="Latitude"
          value={lat}
          icon={
            <svg
              viewBox="0 0 24 24"
              className="size-6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
            >
              <path d="M20 10c0 5-8 12-8 12S4 15 4 10a8 8 0 1 1 16 0Z" />
              <circle cx="12" cy="10" r="2.5" />
            </svg>
          }
        />
        <DetailRow
          label="Longitude"
          value={lon}
          icon={
            <svg
              viewBox="0 0 24 24"
              className="size-6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
            >
              <path d="M20 10c0 5-8 12-8 12S4 15 4 10a8 8 0 1 1 16 0Z" />
              <circle cx="12" cy="10" r="2.5" />
            </svg>
          }
        />
        {isPlane && (
          <DetailRow
            label="Data e hora"
            value={formatTimestamp(selection.plane.time)}
            icon={
              <svg
                viewBox="0 0 24 24"
                className="size-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3 2" />
              </svg>
            }
          />
        )}
        <DetailRow
          label="Aeroporto de origem"
          value={nearby ? nearby.name : "N/A"}
          icon={
            <svg
              viewBox="0 0 24 24"
              className="size-6 rotate-90"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M10.7 3.5 10.8 10 4 13.5v2l6.8-1.6-.8 6.1L12 19l2 1-.8-6.1L20 15.5v-2L13.2 10l.1-6.5Q12 2.1 10.7 3.5Z" />
            </svg>
          }
        />
      </div>
    </aside>
  );
}
