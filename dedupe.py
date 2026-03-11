import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, CACHE_TTL

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def make_fingerprint(x1, y1, x2, y2, camera_id: str):
    return f"{camera_id}:{round(x1/10)}_{round(y1/10)}_{round(x2/10)}_{round(y2/10)}"

def is_duplicate(fingerprint: str) -> bool:
    if r.get(fingerprint):
        return True
    r.setex(fingerprint, CACHE_TTL, "1")
    return False
