from __future__ import annotations
import hashlib,json,re,tempfile
from dataclasses import dataclass
from datetime import UTC,datetime
from pathlib import Path
from typing import Any,Callable,Mapping
import requests
Validator=Callable[[bytes],None]
@dataclass(frozen=True,slots=True)
class CachedResponse:
    path:Path; cache_hit:bool; sha256:str
    @property
    def metadata_path(self)->Path: return self.path.with_suffix(self.path.suffix+".json")
class HttpClient:
    _KEY_PATTERN=re.compile(r"[^A-Za-z0-9._-]+")
    def __init__(self,cache_dir:Path,timeout:float=60.0)->None:
        self.cache_dir=cache_dir; self.timeout=timeout; self.cache_dir.mkdir(parents=True,exist_ok=True); self.session=requests.Session(); self.session.headers.update({"User-Agent":"Prospector/0.3.7 (archaeological research tool)"})
    @staticmethod
    def _request_digest(url:str,params:dict[str,Any]|None)->str:
        return hashlib.sha256(json.dumps([url,params],sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
    @classmethod
    def _safe_key(cls,key:str)->str: return cls._KEY_PATTERN.sub("-",key).strip(".-") or "cache"
    def _cache_path(self,key,url,params,suffix): return self.cache_dir/self._safe_key(key)/f"{self._request_digest(url,params)}{suffix}"
    @staticmethod
    def _sha256(data:bytes)->str: return hashlib.sha256(data).hexdigest()
    @staticmethod
    def _metadata_path(path:Path)->Path: return path.with_suffix(path.suffix+".json")
    @staticmethod
    def _atomic_write(path:Path,data:bytes)->None:
        path.parent.mkdir(parents=True,exist_ok=True); temporary_path=None
        try:
            with tempfile.NamedTemporaryFile(mode="wb",dir=path.parent,prefix=f".{path.name}.",suffix=".part",delete=False) as handle:
                temporary_path=Path(handle.name); handle.write(data); handle.flush()
            temporary_path.replace(path)
        finally:
            if temporary_path is not None: temporary_path.unlink(missing_ok=True)
    def _write_metadata(self,path,*,url,params,sha256,content_type,size_bytes):
        self._atomic_write(self._metadata_path(path),json.dumps({"url":url,"params":params,"retrieved_at":datetime.now(UTC).isoformat(),"sha256":sha256,"size_bytes":size_bytes,"content_type":content_type},indent=2,sort_keys=True).encode())
    @staticmethod
    def _invalidate(path): path.unlink(missing_ok=True); path.with_suffix(path.suffix+".json").unlink(missing_ok=True)
    def _read_cached(self,path,*,validator):
        metadata_path=self._metadata_path(path)
        if not path.is_file() or not metadata_path.is_file(): return None
        try:
            data=path.read_bytes(); metadata=json.loads(metadata_path.read_text(encoding="utf-8")); actual=self._sha256(data)
            if actual!=metadata["sha256"] or metadata.get("size_bytes")!=len(data): raise ValueError
            if validator: validator(data)
        except Exception: self._invalidate(path); return None
        return CachedResponse(path,True,actual)
    def cached_bytes(self,key,url,params=None,*,suffix=".bin",validator=None,headers:Mapping[str,str]|None=None):
        if not suffix.startswith("."): suffix="."+suffix
        path=self._cache_path(key,url,params,suffix); cached=self._read_cached(path,validator=validator)
        if cached: return cached
        response=self.session.get(url,params=params,headers=dict(headers or {}),timeout=self.timeout); response.raise_for_status(); data=response.content
        if not data: raise RuntimeError(f"HTTP response from {url} was empty")
        if validator: validator(data)
        sha=self._sha256(data); self._atomic_write(path,data); self._write_metadata(path,url=url,params=params,sha256=sha,content_type=response.headers.get("content-type"),size_bytes=len(data))
        return CachedResponse(path,False,sha)
    def cached_json(self,key,url,params=None,*,validator=None,headers:Mapping[str,str]|None=None):
        def validate(data):
            parsed=json.loads(data.decode());
            if validator: validator(parsed)
        cached=self.cached_bytes(key,url,params,suffix=".json",validator=validate,headers=headers)
        return json.loads(cached.path.read_text(encoding="utf-8")),cached
