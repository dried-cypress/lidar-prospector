from prospector.domain import Coordinate, StudyArea


def test_study_area_uses_bng_and_contains_centre() -> None:
    area = StudyArea.from_coordinate(Coordinate(50.8657, -0.2405), 1000)
    assert area.crs == "EPSG:27700"
    assert round(area.easting, 2) == 523915.55
    assert round(area.northing, 2) == 108829.39
    xmin, ymin, xmax, ymax = area.bounds
    assert xmin < area.easting < xmax
    assert ymin < area.northing < ymax
    assert 999 < xmax - xmin < 1001
    assert 999 < ymax - ymin < 1001
