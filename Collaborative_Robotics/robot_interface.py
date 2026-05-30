import time


class RobotInterface:
    """Simple robot hand interface stub.

    Replace the `send_action` implementation with your real communication
    channel (serial, USB, network, GPIO, etc.).
    """

    def __init__(self, connection=None):
        self.connection = connection

    def send_action(self, action: str) -> None:
        # Placeholder implementation — replace with real robot commands.
        print(f"[RobotInterface] send_action: {action}")
        # simulate delay
        time.sleep(0.05)
