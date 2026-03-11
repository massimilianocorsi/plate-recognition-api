def classify_plate_type(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    ratio = width / height

    if ratio > 4.0:
        plate_type = "rear_long"
    elif ratio < 2.0:
        plate_type = "rear_square"
    else:
        plate_type = "front"

    return plate_type, ratio


def get_direction(plate_type: str):
    # camera on the left side of the gate
    if plate_type == "front":
        return "ingress"
    return "egress"
