import io
import html
import json
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
)

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from bridge import Bridge
from planes import get_lista


def _get_heading(trajectory):
    if not trajectory or len(trajectory) < 2:
        return 0
#latitude is in [0] and longitude is in [1]
    try:
        current = trajectory[0]
        next_point = trajectory[1]
        lat1 = radians(float(current[0]))
        lon1 = radians(float(current[1]))
        lat2 = radians(float(next_point[0]))
        lon2 = radians(float(next_point[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return 0

    delta_lon = lon2 - lon1
    y = sin(delta_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    bearing = degrees(atan2(y, x))
    return (bearing + 360) % 360


def _build_plane_icon(heading):
    return folium.DivIcon(
        html=(
            f'<div style="font-size:36px; color:#dc2626; '
            f'display:inline-block; transform:rotate({heading}deg);">✈</div>'
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

    def update_map(self):

        m = folium.Map(
            location=[-14.235, -51.925],
            zoom_start=4,
            tiles="Esri.WorldTopoMap"
        )

        planes = get_lista()

        plane_data = [
            {
                "callsign": plane["callsign"],
                "icao": plane["icao"],
                "route_points": [[point["lat"], point["lon"]] for point in plane.get("trajectory", []) if point.get("pred") is False],
                "pred_points": [[point["lat"], point["lon"]] for point in plane.get("trajectory", []) if point.get("pred") is True],
            }
            for plane in planes
        ]
        
        for p in plane_data:
            trajectory = p.get("route_points", []) or []
            if not trajectory:
                continue

            heading = _get_heading(trajectory[-2:])
            folium.Marker(
                location=[p["route_points"][-1][0], p["route_points"][-1][1]],
                tooltip=p["icao"],
                icon=_build_plane_icon(heading),
            ).add_to(m)

            nearby = self.get_nearby_airports(p["route_points"][0][0], p["route_points"][0][1])
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

        function attachClicks(){

            for(let i in window){

                try{

                    let obj = window[i];

                    if(obj && obj._latlng){

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
                                    currentConnectorLine = L.polyline(
                                        [routePoints[routePoints.length - 1], predPoints[0]],
                                        {
                                            color: 'red',
                                            weight: 3,
                                            opacity: 1.0
                                        }
                                    ).addTo(map);
                                    } else {
                                    currentConnectorLine = null;
                                    }

                                // draw predicted points as a grey dotted line if available
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