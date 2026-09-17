def remove_ground(
    xyz,
    intensity,
    ground_threshold
):
    non_ground_mask = xyz[:, 2] > ground_threshold

    non_ground_xyz = xyz[non_ground_mask]
    non_ground_intensity = intensity[non_ground_mask]

    return non_ground_xyz, non_ground_intensity