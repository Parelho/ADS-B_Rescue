import pyModeS as pms
from pyModeS.extra.tcpclient import TcpClient

aircrafts = []
det_coords = [-23.647143, -46.574283]

class Aircraft():
    icao: int
    lat: float
    lon: float
    speed: int
    heading: float
    vertical_rate: int
    speed_type: str

# define your custom class by extending the TcpClient
#   - implement your handle_messages() methods
class ADSBClient(TcpClient):
    def __init__(self, host, port, rawtype):
        super(ADSBClient, self).__init__(host, port, rawtype)

    def handle_messages(self, messages):
        # print(messages)
        for msg, ts in messages:
            msg = msg[12:]
            # print(msg)
            if len(msg) != 28:  # wrong data length
                # print("Data length wrong")
                continue

            df = pms.df(msg)

            if df != 17:  # not ADSB
                print("Not ADS-B")
                continue

            if pms.crc(msg) != 0:  # CRC fail
                print("CRC failed")
                continue

            icao = pms.adsb.icao(msg)
            if icao not in aircrafts:
                aircraft: Aircraft = {
                    "icao": icao,
                    "lat": None,
                    "lon": None,
                    "speed": None,
                    "heading": None,
                    "vertical_rate": None,
                    "speed_type": None
                }
                aircrafts.append(aircraft)

            aircraft = next(item for item in aircrafts if item["icao"] == icao)
            
            tc = pms.adsb.typecode(msg)

            if 5 <= tc <= 18:
                lat, lon = pms.adsb.position_with_ref(msg, det_coords[0], det_coords[1]) # aircraft positon based on base station position
                # print(f"lat: {lat}, lon: {lon}")
                aircraft["lat"] = lat
                aircraft["lon"] = lon
            elif tc == 19:
                velocity = pms.adsb.velocity(msg)
                # print(f"speed: {velocity[0]}, heading: {velocity[1]}, vertical rate: {velocity[2]}, speed type: {velocity[3]}")
                aircraft["speed"] = velocity[0]
                aircraft["heading"] = velocity[1]
                aircraft["vertical_rate"] = velocity[2]
                aircraft["speed_type"] = velocity[3]

            print(aircrafts)

# run new client, change the host, port, and rawtype if needed
client = ADSBClient(host='10.33.133.22', port=30002, rawtype='raw')
client.run()