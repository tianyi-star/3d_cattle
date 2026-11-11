import numpy as np

from cattle3d.cameras import intrinsics, look_at, project_points
from cattle3d.projection import compose_projection, decompose_projection, read_projection_xml, write_projection_xml


def test_look_at_is_a_proper_opencv_camera():
    eye = np.array([0.0, -2.5, 2.0])
    target = np.array([0.0, 0.0, 1.4])
    camera = look_at(eye, target, np.array([0.0, 0.0, 1.0]), 160, 100, intrinsics(160, 100, 180))
    assert np.isclose(np.linalg.det(camera.R), 1.0)
    center = project_points(camera, target[None])
    above = project_points(camera, (target + np.array([0.0, 0.0, 0.3]))[None])
    east = project_points(camera, (target + np.array([0.3, 0.0, 0.0]))[None])
    assert above[1][0] < center[1][0]
    assert east[0][0] > center[0][0]
    assert np.allclose(camera.center(), eye)


def test_projection_roundtrip(tmp_path):
    eye = np.array([2.0, -1.5, 3.0])
    target = np.array([0.2, 0.4, 1.2])
    K = intrinsics(320, 200, 280)
    camera = look_at(eye, target, np.array([0.0, 0.0, 1.0]), 320, 200, K, camera_id=2)
    projection = compose_projection(K, camera.R, camera.t)
    recovered_k, rotation, translation = decompose_projection(projection)
    point = np.array([[0.4, -0.2, 1.6]])
    original = project_points(camera, point)
    from cattle3d.cameras import Camera

    restored = Camera(rotation, translation, recovered_k, 320, 200)
    again = project_points(restored, point)
    assert np.allclose(original[0], again[0], atol=1e-4)
    assert np.allclose(original[1], again[1], atol=1e-4)
    path = tmp_path / "proj_mat_cam2.xml"
    write_projection_xml(path, projection, K)
    loaded, stored_k = read_projection_xml(path)
    assert np.allclose(loaded, projection)
    assert np.allclose(stored_k, K)
