from __future__ import annotations
import hashlib,json,platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from prospector import __version__
from prospector.config import AppConfig
from prospector.domain import Run,StudyArea
RUN_DIRECTORIES=("inputs","sources","terrain","overlays","candidates","reports","logs")
def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
def _write_json_atomic(path:Path,data:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(data,indent=2,default=str)+"\n",encoding="utf-8"); temporary.replace(path)
def create_run(config:AppConfig,area:StudyArea)->Run:
    now=datetime.now(timezone.utc); base_id=now.strftime("%Y%m%dT%H%M%SZ"); suffix=0
    while True:
        run_id=f"{base_id}{f'-{suffix:02d}' if suffix else ''}"; path=config.runs_dir/run_id
        if not path.exists(): break
        suffix+=1
    path.mkdir(parents=True)
    for name in RUN_DIRECTORIES:(path/name).mkdir()
    run=Run(run_id,area,now,__version__,str(path))
    _write_json_atomic(path/"run.json",{"run_id":run.run_id,"application_version":run.application_version,"started_at":run.started_at.isoformat(),"status":"running","runtime":{"python_version":platform.python_version(),"platform":platform.platform()},"parameters":{"latitude":area.centre.latitude,"longitude":area.centre.longitude,"diameter_m":area.diameter_m,"crs":area.crs},"area":{"latitude":area.centre.latitude,"longitude":area.centre.longitude,"diameter_m":area.diameter_m,"easting":area.easting,"northing":area.northing,"crs":area.crs,"geometry_wkt":area.geometry_wkt},"analysis":{"detector":{"name":None,"version":None,"status":"not implemented"},"classifier":{"name":None,"version":None,"status":"not implemented"}},"sources":{},"outputs":{},"errors":[]})
    return run
def update_run_metadata(run:Run,*,status=None,sources=None,outputs=None,errors=None,analysis=None,finished=False)->None:
    path=Path(run.path)/"run.json"; metadata=json.loads(path.read_text(encoding="utf-8"))
    if status is not None: metadata["status"]=status
    if sources: metadata.setdefault("sources",{}).update(sources)
    if analysis: metadata.setdefault("analysis",{}).update(analysis)
    if outputs: metadata.setdefault("outputs",{}).update(outputs)
    if errors: metadata.setdefault("errors",[]).extend(errors)
    if finished: metadata["finished_at"]=datetime.now(timezone.utc).isoformat()
    _write_json_atomic(path,metadata)
