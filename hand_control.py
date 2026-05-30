import time
import cv2
from cv import ObjectDetector
from robot_interface import RobotInterface


ROBOT_ACTIONS = {
    'bottle': 'open_hand',
    'cup': 'open_hand',
    'cell phone': 'point',
    'book': 'grasp',
    'remote': 'hold',
    'keyboard': 'rest',
    'mouse': 'rest',
    'banana': 'grasp',
    'apple': 'grasp',
    'laptop': 'point',
}


def control_robot_hand(robot: RobotInterface, action: str):
    # Send action to the provided robot interface.
    robot.send_action(action)


def choose_action(detections):
    for det in detections:
        label = det['label']
        if label in ROBOT_ACTIONS:
            return ROBOT_ACTIONS[label], label
    return 'idle', None


def main():
    detector = ObjectDetector(conf_thres=0.4)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam. Check your camera settings.')

    current_action = None
    robot = RobotInterface()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        action, label = choose_action(detections)

        if action != current_action:
            current_action = action
            if label:
                print(f'Detected object: {label} -> action: {action}')
            control_robot_hand(robot, action)

        frame = detector.draw_boxes(frame, detections)
        cv2.putText(frame, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow('Robot Hand Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
