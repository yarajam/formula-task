def extract_roi(xyz,intensity,
                x_min,x_max,
                y_min,y_max,
                z_min,z_max):
    
    roi_mask = (
        (xyz[:, 0] >= x_min) &
        (xyz[:, 0] <= x_max) &
        (xyz[:, 1] >= y_min) &
        (xyz[:, 1] <= y_max) &
        (xyz[:, 2] >= z_min) &
        (xyz[:, 2] <= z_max)
    )

    roi_xyz = xyz[roi_mask]
    roi_intensity = intensity[roi_mask]

    return roi_xyz, roi_intensity