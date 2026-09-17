import numpy as np


def transform_points(xyz, tf_transform):

    t = tf_transform.transform.translation
    q = tf_transform.transform.rotation

    rotation_matrix = np.array([
        [
            1 - 2 * (q.y**2 + q.z**2),
            2 * (q.x*q.y - q.z*q.w),
            2 * (q.x*q.z + q.y*q.w)
        ],
        [
            2 * (q.x*q.y + q.z*q.w),
            1 - 2 * (q.x**2 + q.z**2),
            2 * (q.y*q.z - q.x*q.w)
        ],
        [
            2 * (q.x*q.z - q.y*q.w),
            2 * (q.y*q.z + q.x*q.w),
            1 - 2 * (q.x**2 + q.y**2)
        ]
    ])

    translation = np.array([
        t.x,
        t.y,
        t.z
    ])

    return xyz @ rotation_matrix.T + translation