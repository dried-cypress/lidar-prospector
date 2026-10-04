from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, Field
class AppConfig(BaseModel):
    application_name: str = "Prospector"
    data_dir: Path
    runs_dir: Path
    cache_dir: Path
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    default_diameter_m: float = Field(default=1000.0, gt=0, le=10000)
    @classmethod
    def for_project(cls, project: Path) -> "AppConfig":
        project = project.resolve()
        return cls(data_dir=project/"data", runs_dir=project/"runs", cache_dir=project/"data"/"cache")
    def ensure_directories(self) -> None:
        for path in (self.data_dir,self.data_dir/"raw",self.data_dir/"derived",self.data_dir/"projects",self.cache_dir,self.runs_dir): path.mkdir(parents=True,exist_ok=True)
