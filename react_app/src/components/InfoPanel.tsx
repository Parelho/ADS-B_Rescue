import type { Airport, Selection } from "../types";
import { findNearbyAirport } from "../lib/airports";
import { formatTimestamp } from "../lib/format";

interface Props {
  selection: Selection | null;
  airports: Airport[];
  refreshing: boolean;
}

export default function InfoPanel({ selection, airports, refreshing }: Props) {
  if (refreshing) {
    return <aside className="info-panel">Atualizando dados da API...</aside>;
  }
  if (!selection) {
    return (
      <aside className="info-panel">
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
    <aside className="info-panel">
      <div className="info-title">{title}</div>
      <hr />
      <div className="info-body">
        <b>Latitude:</b> {lat}
        <br />
        <b>Longitude:</b> {lon}
        <br />
        {isPlane && (
          <>
            <b>Data e hora:</b> {formatTimestamp(selection.plane.time)}
            <br />
          </>
        )}
        <b>Aeroporto de origem:</b> {nearby ? nearby.name : "N/A"}
      </div>
    </aside>
  );
}
