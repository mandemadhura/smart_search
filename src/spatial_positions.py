def calculate_spatial_position(bbox, image_shape):
    """
    Calculate spatial position relative to camera view
    """
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox

    # Center of object
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Normalize to -1 to 1 range
    norm_x = (center_x - width/2) / (width/2)
    norm_y = (center_y - height/2) / (height/2)

    # Determine position descriptions
    horizontal_pos = "center"
    if norm_x < -0.3:
        horizontal_pos = "left"
    elif norm_x > 0.3:
        horizontal_pos = "right"

    vertical_pos = "eye level"
    if norm_y < -0.3:
        vertical_pos = "above"
    elif norm_y > 0.3:
        vertical_pos = "below"

    return horizontal_pos, vertical_pos, norm_x, norm_y