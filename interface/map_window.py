import io
import html
import json
from datetime import datetime, timezone
from math import atan2, cos, degrees, radians, sin

import folium
from folium import Element

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QTextEdit,
    QHBoxLayout,
)

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from bridge import Bridge
from planes import get_lista


def _get_heading(trajectory):
    if not trajectory or len(trajectory) < 2:
        return 0

    try:
        current = trajectory[0]
        next_point = trajectory[1]
        lat1 = radians(float(current["lat"]))
        lon1 = radians(float(current["lon"]))
        lat2 = radians(float(next_point["lat"]))
        lon2 = radians(float(next_point["lon"]))
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


class MapWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Aviões em tempo real (OpenSky)")
        self.setGeometry(100, 100, 1100, 650)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QHBoxLayout(central_widget)

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

        layout.addWidget(self.view)
        layout.addWidget(self.info_panel)


        self.bridge = Bridge()
        self.bridge.planeClicked.connect(self.update_panel)

        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

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

    def update_panel(self, callsign, lat, lon, time):
        safe_callsign = html.unescape(callsign).replace('<div>', '').replace('</div>', '')
        safe_time = html.escape(self._format_timestamp(time))
        self.info_panel.setHtml(
            f"<div style='font-size:14pt; font-weight:bold; margin-bottom:8px;'>{safe_callsign}</div>"
            f"<hr style='border:none; border-top:1px solid #d1d5db; margin:0 0 12px 0;'/>"
            f"<div style='font-size:11pt; line-height:1.5;'>"
            f"<b>Latitude:</b> {lat}<br/>"
            f"<b>Longitude:</b> {lon}<br/>"
            f"<b>Data e hora:</b> {safe_time}"
            f"</div>"
        )

    def update_map(self):

        m = folium.Map(
            location=[-14.235, -51.925],
            zoom_start=4,
            tiles="Esri.WorldTopoMap"
        )

        planes = get_lista()

        for p in planes:
            trajectory = p.get("trajectory", []) or []
            heading = _get_heading(trajectory)
            folium.Marker(
                location=[trajectory[0]["lat"], trajectory[0]["lon"]],
                tooltip=p["icao"],
                icon=_build_plane_icon(heading),
            ).add_to(m)

        plane_data = [
            {
                "callsign": plane["callsign"],
                "lat": plane["trajectory"][0]["lat"],
                "lon": plane["trajectory"][0]["lon"],
                "time": plane["trajectory"][0]["timestamp"],
                "route_points": [[point["lat"], point["lon"]] for point in plane.get("trajectory", [])],
            }
            for plane in planes
        ]

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

        function normalizeText(value) {
            return String(value || "")
                .replace(/<[^>]*>/g, "")
                .replace(/&nbsp;/g, " ")
                .trim();
        }

        function findPlaneMatch(tooltip, latlng) {
            const normalizedTooltip = normalizeText(tooltip);

            const byCallsign = planeData.find(function(item) {
                return normalizeText(item.callsign) === normalizedTooltip;
            });

            if (byCallsign) {
                return byCallsign;
            }

            return planeData.find(function(item) {
                return Math.abs(item.lat - latlng.lat) < 0.0001 && Math.abs(item.lon - latlng.lng) < 0.0001;
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

                            if (map) {
                                if (currentLine) {
                                    map.removeLayer(currentLine);
                                }

                                currentLine = L.polyline(
                                    routePoints.concat([[latlng.lat, latlng.lng]]),
                                    {
                                        color: 'blue',
                                        weight: 5,
                                        opacity: 0.8
                                    }
                                ).addTo(map);

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