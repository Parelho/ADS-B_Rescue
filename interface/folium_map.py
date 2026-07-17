import io
from datetime import datetime, timezone
from math import atan2, cos, degrees, radians, sin

import folium
from folium import Element
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


def format_timestamp(value):
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

def build_map_html():
    m = folium.Map(
        location=[-14.235, -51.925],
        zoom_start=4,
        tiles="Esri.WorldTopoMap"
    )

    planes = get_lista()

    for p in planes:
        folium.Marker(
            location=[p["trajectory"][0]["lat"], p["trajectory"][0]["lon"]],
            tooltip=p["icao"],
            popup=(
                f"<b>{p['icao']}</b><br/>"
                f"Callsign: {p['callsign']}<br/>"
                f"Lat: {p['trajectory'][0]['lat']:.4f}<br/>"
                f"Lon: {p['trajectory'][0]['lon']:.4f}"
            ),
            icon=folium.Icon(color="red", icon="plane", prefix="fa")
        ).add_to(m)

    channel_js = """
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script>
    new QWebChannel(qt.webChannelTransport, function(channel) {
        window.bridge = channel.objects.bridge;
    });
    </script>
    """

    click_js = """
    <script>
    function attachClicks() {
        for (let i in window) {
            try {
                let obj = window[i];
                if (obj && obj._latlng) {
                    obj.on('click', function(e) {
                        let p = e.target;
                        window.bridge.onPlaneClicked(
                            p.getTooltip() ? p.getTooltip().getContent() : "N/A",
                            "Desconhecido",
                            p.getLatLng().lat,
                            p.getLatLng().lng
                        );
                    });
                }
            } catch (e) {}
        }
    }

    setTimeout(attachClicks, 1500);
    </script>
    """

    m.get_root().html.add_child(Element(channel_js))
    m.get_root().html.add_child(Element(click_js))

    data = io.BytesIO()
    m.save(data, close_file=False)

    return data.getvalue().decode()


def build_folium_map(planes=None):
    if planes is None:
        planes = get_lista()

    m = folium.Map(
        location=[-14.235, -51.925],
        zoom_start=4,
        tiles="Esri.WorldTopoMap"
    )

    for plane in planes:
        trajectory = plane.get("trajectory", []) or []
        heading = _get_heading(trajectory)
        popup_html = (
            f"<b>{plane['callsign']}</b><br/>"
            f"País: {plane['country']}<br/>"
            f"Lat: {plane['lat']:.4f}<br/>"
            f"Lon: {plane['lon']:.4f}"
        )
        folium.Marker(
            location=[plane["lat"], plane["lon"]],
            tooltip=plane["callsign"],
            popup=popup_html,
            icon=_build_plane_icon(heading)
        ).add_to(m)

    return m


def run_streamlit_app():
    try:
        import streamlit as st
        from streamlit_folium import st_folium
    except ImportError as exc:
        raise SystemExit(
            "Instale streamlit e streamlit-folium para usar esta visualização: "
            "pip install streamlit streamlit-folium"
        ) from exc

    st.set_page_config(page_title="Aviões em tempo real", layout="wide")
    st.title("Aviões em tempo real (OpenSky)")

    planes = get_lista()

    if not planes:
        st.warning("Nenhum avião encontrado no momento.")
        return

    selected_plane = st.session_state.get("selected_plane")
    if selected_plane is None:
        selected_plane = planes[0]

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Detalhes do voo")
        if selected_plane:
            st.metric("Callsign", selected_plane["callsign"])
            st.write(f"País: {selected_plane['country']}")
            st.write(f"Latitude: {selected_plane['lat']:.4f}")
            st.write(f"Longitude: {selected_plane['lon']:.4f}")
            st.write(f"Hora: {format_timestamp(selected_plane.get('hora'))}")
        else:
            st.info("Clique em um marcador para ver os detalhes.")

    with right_col:
        map_obj = build_folium_map(planes)
        try:
            map_state = st_folium(
                map_obj,
                width=900,
                height=600,
                key="plane-map",
                returned_objects=["last_object_clicked", "bounds"],
            )
        except TypeError:
            map_state = st_folium(map_obj, width=900, height=600, key="plane-map")

        clicked = map_state.get("last_object_clicked") or {}
        tooltip = clicked.get("tooltip")

        if tooltip:
            match = next((plane for plane in planes if plane["callsign"] == tooltip), None)
            if match:
                st.session_state["selected_plane"] = match

        if selected_plane and selected_plane not in planes:
            st.session_state["selected_plane"] = planes[0]