def test_base():
    from d3nav.base import NAME

    assert NAME == "d3nav"


def test_import():
    from d3nav import load_d3nav
    from d3nav import center_crop, d3nav_transform_img
    from d3nav import visualize_frame_img
