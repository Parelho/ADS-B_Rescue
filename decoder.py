import pyModeS as pms
from pyModeS.extra.tcpclient import TcpClient

adsb_dict = []
det_coords = [-23.647143, -46.574283]

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

          if pms.crc(msg) !=0:  # CRC fail
              print("CRC failed")
              continue

          icao = pms.adsb.icao(msg)
          tc = pms.adsb.typecode(msg)

          # TODO: write you magic code here
          # print(ts, icao, tc, msg)
          if 5 <= tc <= 18:
            print(pms.adsb.position_with_ref(msg, det_coords[0], det_coords[1]))


# run new client, change the host, port, and rawtype if needed
client = ADSBClient(host='10.33.133.22', port=30002, rawtype='raw')
client.run()