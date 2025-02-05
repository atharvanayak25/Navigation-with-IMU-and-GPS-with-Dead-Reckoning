import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import serial
import utm
import sys
from gps_msg.msg import GPSmsg

class GPSDriver(Node):
    def __init__(self):
        super().__init__('gpsDriver')
        self.declare_parameter('port', '/dev/pts/4')
        self.declare_parameter('baudrate', 4800)

        serial_port = self.get_parameter('port').value
        serial_baud = self.get_parameter('baudrate').value

        self.port = serial.Serial(serial_port, serial_baud, timeout=3.0)
        self.publisher = self.create_publisher(GPSmsg, '/gps', 10)
        self.timer = self.create_timer(0.01, self.read_data)

    def read_data(self):
        data = self.port.readline().decode().strip()
        if "$GPGGA" in data:
            self.parse_gpgga(data)

    def convert_to_decimal(self, co_ord, direction):
        decimal_deg = int(co_ord/100)
        decimal_min = float(co_ord) - (decimal_deg*100)
        converted_co_ord = float(decimal_deg + decimal_min/60)

        if direction == 'S' or direction == 'W':
            converted_co_ord = -converted_co_ord

        return converted_co_ord

    def parse_gpgga(self, data):
        parts = data.split(',')
        if parts[0] != "$GPGGA":
            return None
        lat = self.convert_to_decimal(float(parts[2]),parts[3])
        lon = self.convert_to_decimal(float(parts[4]), parts[5])
        alt = float(parts[9])
        utm_easting, utm_northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        now = self.get_clock().now()
        secs = now.seconds_nanoseconds()[0]
        nanosecs = now.seconds_nanoseconds()[1]
        msg = GPSmsg()
        msg.header.stamp.sec = secs
        msg.header.stamp.nanosec = nanosecs
        msg.header.frame_id = 'GPS1_Frame'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = alt
        msg.utm_easting = utm_easting
        msg.utm_northing = utm_northing
        msg.zone = str(zone_number)
        msg.letter = zone_letter

        self.publisher.publish(msg)
        self.get_logger().info(f'Current GPS status: {msg}')

def main(args=None):
    rclpy.init(args=args)
    gps_driver = GPSDriver()

    try:
        rclpy.spin(gps_driver)
    except KeyboardInterrupt:
        pass
    finally:
        gps_driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()