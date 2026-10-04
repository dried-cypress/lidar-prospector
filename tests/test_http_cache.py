from pathlib import Path
from prospector.providers.http import HttpClient
class FakeResponse:
    def __init__(self,content): self.content=content; self.headers={"content-type":"application/octet-stream"}
    def raise_for_status(self): pass
class FakeSession:
    def __init__(self,content=b"cached test data"): self.calls=0; self.headers={}; self.content=content
    def get(self,url,params=None,headers=None,timeout=None): self.calls+=1; return FakeResponse(self.content)
def test_cached_bytes_only_downloads_once(tmp_path:Path):
    client=HttpClient(tmp_path); fake=FakeSession(); client.session=fake
    first=client.cached_bytes("test","https://example.test/data",{"a":"1"}); second=client.cached_bytes("test","https://example.test/data",{"a":"1"})
    assert not first.cache_hit and second.cache_hit and fake.calls==1
