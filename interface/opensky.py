import requests

OPEN_SKY_URL = "https://opensky-network.org/api/states/all"


def get_planes():
    try:
        r = requests.get(OPEN_SKY_URL, timeout=10)
        data = r.json()

        planes = []

        for state in data.get("states", []):
            callsign = state[1]
            country = state[2]
            lon = state[5]
            lat = state[6]

            if lat is None or lon is None:
                continue

            # filtro Brasil
            if -34 <= lat <= 6 and -75 <= lon <= -30:
                planes.append({
                    "callsign": callsign.strip() if callsign else "N/A",
                    "lat": lat,
                    "lon": lon,
                    "country": country
                })

        return planes

    except Exception as e:
        print("Erro OpenSky:", e)
        return []