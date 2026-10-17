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
      <div className="space-y-2 text-[15px] leading-relaxed">
        <div>
          <b>Latitude:</b> {lat}
        </div>
        <div>
          <b>Longitude:</b> {lon}
        </div>
        {isPlane && (
          <div>
            <b>Data e hora:</b> {formatTimestamp(selection.plane.time)}
          </div>
        )}
        <div>
          <b>Aeroporto de origem:</b> {nearby ? nearby.name : "N/A"}
        </div>
      </div>
    </aside>
  );
}
