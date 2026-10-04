import json
from pathlib import Path
from prospector.config import AppConfig
from prospector.domain import Coordinate,StudyArea
from prospector.runs import create_run,sha256_file,update_run_metadata
def test_create_run_creates_expected_structure(tmp_path:Path):
    config=AppConfig.for_project(tmp_path/"project"); config.ensure_directories(); run=create_run(config,StudyArea.from_coordinate(Coordinate(50.8657,-0.2405),1000)); p=Path(run.path)
    assert {x.name for x in p.iterdir() if x.is_dir()}=={"inputs","sources","terrain","overlays","candidates","reports","logs"}; assert (p/"run.json").is_file()
def test_update_run_metadata(tmp_path:Path):
    config=AppConfig.for_project(tmp_path/"project"); config.ensure_directories(); run=create_run(config,StudyArea.from_coordinate(Coordinate(50.8657,-0.2405),1000)); update_run_metadata(run,status="completed",outputs={"report":"reports/report.html"},finished=True); data=json.loads((Path(run.path)/"run.json").read_text()); assert data["status"]=="completed" and "finished_at" in data
def test_sha256_file(tmp_path:Path):
    p=tmp_path/"x"; p.write_bytes(b"hello"); assert sha256_file(p)=="2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
