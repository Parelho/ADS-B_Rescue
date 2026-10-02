import io
import html
import json
import math
import pandas as pd
from datetime import datetime, timezone
from math import atan2, cos, degrees, radians, sin

import folium
from folium import Element

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QCompleter,
)
from PyQt6.QtCore import QStringListModel, Qt

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from bridge import Bridge
from planes import get_lista


def _get_heading(trajectory):
    if not trajectory or len(trajectory) < 2:
        return 0
#latitude is in [0] and longitude is in [1]
    try:
        previous = trajectory[-2]
        current = trajectory[-1]
        lat1 = radians(float(previous[0]))
        lon1 = radians(float(previous[1]))
        lat2 = radians(float(current[0]))
        lon2 = radians(float(current[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return 0

    delta_lon = lon2 - lon1
    y = sin(delta_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    bearing = degrees(atan2(y, x))
    return (bearing + 360) % 360


def _build_plane_icon(heading):
    icon_rotation = (heading - 90) % 360
    return folium.DivIcon(
        html=(
            f'<div style="width:36px; height:36px; line-height:36px; '
            f'text-align:center; font-size:36px; color:#dc2626; '
            f'display:block; transform:rotate({icon_rotation}deg); '
            f'transform-origin:18px 18px;">✈</div>'
        ),
        icon_size=(36, 36),
        icon_anchor=(18, 18),
    )


def _build_airport_icon():
    return folium.DivIcon(
        html=(
            '<div style="font-size:24px; color:#3b82f6; '
            'display:inline-block;">✈</div>'
        ),
        icon_size=(24, 24),
        icon_anchor=(12, 12),
    )


def _make_circle(lat, lon, radius_deg=0.08, points=72):
    """Generate circle coordinates around a point for map visualization."""
    coords = []
    for i in range(points + 1):
        angle = 2 * 3.14159265359 * i / points
        dlat = radius_deg * sin(angle)
        dlon = (
            radius_deg * cos(angle)
            / max(cos(radians(lat)), 1e-6)
        )
        coords.append([lat + dlat, lon + dlon])
    return coords


class MapWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aviões em tempo real (OpenSky)")
        self.setGeometry(100, 100, 1100, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)

        # Refresh button on the left
        self.refresh_button = QPushButton("Atualizar dados")
        self.refresh_button.setMinimumHeight(50)
        self.refresh_button.setMinimumWidth(190)
        self.refresh_button.setStyleSheet(
            "QPushButton {"
            "background-color: #ffffff;"
            "color: #000000;"
            "border: 1px solid #000000;"
            "border-radius: 18px;"
            "padding: 12px 22px;"
            "font-size: 13pt;"
            "font-weight: 700;"
            "}"
            "QPushButton:hover {"
            "background-color: #f3f4f6;"
            "border-color: #000000;"
            "}"
            "QPushButton:pressed {"
            "background-color: #e5e7eb;"
            "border-color: #000000;"
            "}"
            "QPushButton:disabled {"
            "background-color: #f3f4f6;"
            "border-color: #000000;"
            "color: #000000;"
            "}"
        )
        self.refresh_button.clicked.connect(self.refresh_map_data)
        toolbar_layout.addWidget(self.refresh_button)

        toolbar_layout.addStretch()

        # Search bar for planes in the center
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Digite o código ICAO")
        self.search_input.setMaximumWidth(300)
        self.search_input.setMinimumHeight(40)
        self.search_input.setStyleSheet(
            "QLineEdit {"
            "background-color: #ffffff;"
            "color: #000000;"
            "border: 2px solid #d1d5db;"
            "border-radius: 8px;"
            "padding: 8px 12px;"
            "font-size: 11pt;"
            "}"
            "QLineEdit:focus {"
            "border: 2px solid #3b82f6;"
            "background-color: #f9fafb;"
            "}"
        )
        self.search_model = QStringListModel(self)
        self.search_completer = QCompleter(self.search_model, self)
        self.search_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.search_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.search_input.setCompleter(self.search_completer)
        self.search_completer.activated[str].connect(self.select_plane)
        toolbar_layout.addWidget(self.search_input)

        toolbar_layout.addStretch()

        main_layout.addWidget(toolbar)

        content_layout = QHBoxLayout()

        self.info_panel = QTextEdit()
        self.info_panel.setFixedWidth(320)
        self.info_panel.setReadOnly(True)
        self.info_panel.setPlainText("Clique em um marcador para ver os detalhes do voo.")
        self.info_panel.setStyleSheet(
            "QTextEdit {"
            "background: #f4f6f8;"
            "border: 1px solid #d1d5db;"
            "border-radius: 10px;"
            "padding: 12px;"
            "font-family: Arial, sans-serif;"
            "font-size: 12pt;"
            "color: #1f2937;"
            "}"
        )

        self.view = QWebEngineView()

        content_layout.addWidget(self.view)
        content_layout.addWidget(self.info_panel)
        main_layout.addLayout(content_layout)

        self.bridge = Bridge()
        self.bridge.planeClicked.connect(self.update_panel)

        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        self.airports = pd.read_csv("airports.csv")

        # apenas aeroportos
        self.airports = self.airports[
            self.airports["type"].isin([
                "small_airport",
                "medium_airport",
                "large_airport"
            ])
        ]

        # Store plane data for the search suggestions.
        self.plane_data = []
        self.update_map()

    @staticmethod
    def _format_timestamp(value):
        if value in (None, "", "N/A"):
            return "N/A"

        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return str(value)

        if timestamp > 1e12:
            timestamp = timestamp / 1000.0

        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC")
        except (OverflowError, OSError, ValueError):
            return str(value)

    def get_nearby_airports(self, lat, lon, delta=0.2):

        nearby = self.airports[
        (self.airports["latitude_deg"].between(lat - delta, lat + delta)) &
        (self.airports["longitude_deg"].between(lon - delta, lon + delta))
    ]

        return nearby

    def update_panel(self, callsign, lat, lon, time):
        safe_callsign = html.unescape(callsign).replace('<div>', '').replace('</div>', '')
        airports = self.get_nearby_airports(lat, lon)
        safe_time = html.escape(self._format_timestamp(time))
        self.info_panel.setHtml(
            f"<div style='font-size:14pt; font-weight:bold; margin-bottom:8px;'>{safe_callsign}</div>"
            f"<hr style='border:none; border-top:1px solid #d1d5db; margin:0 0 12px 0;'/>"
            f"<div style='font-size:11pt; line-height:1.5;'>"
            f"<b>Latitude:</b> {lat}<br/>"
            f"<b>Longitude:</b> {lon}<br/>"
            f"<b>Data e hora:</b> {safe_time}<br/>"
            f"<b>Aeroporto de origem:</b> {airports.iloc[0]['name'] if not airports.empty else 'N/A'}"
            f"</div>"
        )

    def refresh_map_data(self):
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("Atualizando...")
        self.info_panel.setPlainText("Atualizando dados da API...")
        self.update_map()
        self.refresh_button.setEnabled(True)
        self.refresh_button.setText("Atualizar dados")

    def select_plane(self, icao):
        """Center the map and trigger the selected plane marker click."""
        self.view.page().runJavaScript(
            "window.selectPlaneByIcao(%s);" % json.dumps(icao)
        )

    def update_map(self):

        m = folium.Map(
            location=[-14.235, -51.925],
            zoom_start=4,
            tiles="Esri.WorldTopoMap"
        )

        planes = get_lista()

        plane_data = []
        for plane in planes:
            trajectory = plane.get("trajectory", []) or []
            route_trajectory = [
                point for point in trajectory if point.get("pred") is False
            ]
            plane_data.append({
                "callsign": plane["callsign"],
                "icao": plane["icao"],
                "time": route_trajectory[-1].get("timestamp") if route_trajectory else None,
                "route_points": [
                    [point["lat"], point["lon"]]
                    for point in route_trajectory
                ],
                "pred_points": [
                    [point["lat"], point["lon"]]
                    for point in trajectory if point.get("pred") is True
                ],
            })

        # Store all plane data for later reference
        self.plane_data = plane_data
        self.search_model.setStringList([
            str(plane["icao"]).upper()
            for plane in plane_data
            if plane.get("icao")
        ])

        for p in plane_data:
            route_points = p.get("route_points", []) or []
            if route_points:
                last_route_lat = route_points[-1][0]
                last_route_lon = route_points[-1][1]
                p["search_area"] = _make_circle(last_route_lat, last_route_lon, radius_deg=0.08)
            else:
                p["search_area"] = None

        for p in plane_data:
            route_points = p.get("route_points", []) or []
            if not route_points:
                continue

            heading = _get_heading(route_points)
            folium.Marker(
                location=[route_points[-1][0], route_points[-1][1]],
                tooltip=p["icao"],
                icon=_build_plane_icon(heading),
            ).add_to(m)

            nearby = self.get_nearby_airports(route_points[0][0], route_points[0][1])
            if not nearby.empty:
                airport = nearby.iloc[0]
                folium.Marker(
                    location=[float(airport["latitude_deg"]), float(airport["longitude_deg"])],
                    tooltip=airport["name"],
                    icon=_build_airport_icon(),
                ).add_to(m)



        channel_js = """
        <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
        <script>
        new QWebChannel(qt.webChannelTransport, function(channel) {
            window.bridge = channel.objects.bridge;
        });
        </script>
        """

        m.get_root().html.add_child(Element(channel_js))

        click_js = """
        <script>
        let planeData = """ + json.dumps(plane_data) + """;
        let currentLine = null;
        let currentPredLine = null;
        let currentConnectorLine = null;
        let currentSearchArea = null;

        function normalizeText(value) {
            return String(value || "")
                .replace(/<[^>]*>/g, "")
                .replace(/&nbsp;/g, " ")
                .trim();
        }

        function findPlaneMatch(tooltip, latlng) {
            const normalizedTooltip = normalizeText(tooltip);

            const byIcao = planeData.find(function(item) {
                return normalizeText(item.icao) === normalizedTooltip;
            });

            if (byIcao) {
                return byIcao;
            }

            const byCallsign = planeData.find(function(item) {
                return normalizeText(item.callsign) === normalizedTooltip;
            });

            if (byCallsign) {
                return byCallsign;
            }

            return planeData.find(function(item) {
                const lastRoutePoint = item.route_points && item.route_points[item.route_points.length - 1];
                return lastRoutePoint &&
                    Math.abs(lastRoutePoint[0] - latlng.lat) < 0.0001 &&
                    Math.abs(lastRoutePoint[1] - latlng.lng) < 0.0001;
            }) || null;
        }

        window.selectPlaneByIcao = function(icao) {
            const normalizedIcao = normalizeText(icao).toUpperCase();
            attachClicks();

            for (let key in window) {
                try {
                    let marker = window[key];
                    let tooltip = marker && marker.getTooltip ? marker.getTooltip() : null;

                    if (
                        marker &&
                        marker._latlng &&
                        tooltip &&
                        normalizeText(tooltip.getContent()).toUpperCase() === normalizedIcao
                    ) {
                        let map = marker._map;
                        if (map) {
                            map.setView(
                                [marker._latlng.lat, marker._latlng.lng],
                                Math.max(map.getZoom(), 6),
                                {animate: true, duration: 0.7}
                            );
                        }
                        marker.fire('click');
                        return true;
                    }
                } catch (e) {}
            }

            return false;
        };

        function attachClicks(){

            for(let i in window){

                try{

                    let obj = window[i];

                    if(obj && obj._latlng && !obj._planeClickAttached){

                        obj._planeClickAttached = true;

                        obj.on('click', function(e){

                            let p = e.target;
                            let map = p._map;
                            let latlng = p.getLatLng();
                            let tooltip = p.getTooltip() ? p.getTooltip().getContent() : "N/A";
                            let matchedPlane = findPlaneMatch(tooltip, latlng);
                            let routePoints = matchedPlane ? matchedPlane.route_points : [];
                            let predPoints = matchedPlane ? matchedPlane.pred_points : [];

                            if (map) {
                                if (currentLine) {
                                    map.removeLayer(currentLine);
                                }
                                if (currentPredLine) {
                                    map.removeLayer(currentPredLine);
                                }
                                if (currentConnectorLine) {
                                    map.removeLayer(currentConnectorLine);
                                }
                                if (currentSearchArea) {
                                    map.removeLayer(currentSearchArea);
                                }

                                currentLine = L.polyline(
                                    routePoints,
                                    {
                                        color: 'blue',
                                        weight: 5,
                                        opacity: 0.8
                                    }
                                ).addTo(map);

                                if (
                                    routePoints &&
                                    predPoints &&
                                    routePoints.length > 0 &&
                                    predPoints.length > 0
                                ) {
                                    currentConnectorLine = null
                                    } else {
                                    currentConnectorLine = null;
                                    }

                                if (predPoints && predPoints.length > 0) {
                                    currentPredLine = L.polyline(
                                        predPoints,
                                        {
                                            color: '#6b7280',
                                            weight: 3,
                                            opacity: 0.9,
                                            dashArray: '6, 8'
                                        }
                                    ).addTo(map);
                                } else {
                                    currentPredLine = null;
                                }

                                if (matchedPlane && matchedPlane.search_area && matchedPlane.search_area.length > 0) {
                                    currentSearchArea = L.polygon(
                                        matchedPlane.search_area,
                                        {
                                            color: '#3b82f6',
                                            fillColor: '#3b82f6',
                                            fillOpacity: 0.15,
                                            weight: 2
                                        }
                                    ).addTo(map);
                                } else {
                                    currentSearchArea = null;
                                }

                                if (map.getZoom() !== 6) {
                                    map.setView([latlng.lat, latlng.lng], map.getZoom(), {animate: true, duration: 0.7});
                                } else {
                                    map.panTo([latlng.lat, latlng.lng], {animate: true, duration: 0.7});
                                }
                            }

                            window.bridge.onPlaneClicked(
                                tooltip,

                                "Desconhecido",

                                latlng.lat,
                                latlng.lng,
                                matchedPlane && matchedPlane.time !== undefined ? matchedPlane.time : "N/A"
                            );

                        });

                    }

                }catch(e){}

            }

        }

        setTimeout(attachClicks,1500);

        </script>
        """

        m.get_root().html.add_child(Element(click_js))

        data = io.BytesIO()
        m.save(data, close_file=False)

        self.view.setHtml(data.getvalue().decode())