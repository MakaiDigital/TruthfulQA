import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from cachetools import TTLCache
from diskcache import Cache


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass

    @abstractmethod
    def delete(self, key: str):
        pass

    @abstractmethod
    def clear(self):
        pass


class NullCacheBackend(CacheBackend):
    """No-op cache backend when caching is disabled"""

    def get(self, key: str) -> Optional[Any]:
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass

    def delete(self, key: str):
        pass

    def clear(self):
        pass


class DiskCacheBackend(CacheBackend):
    def __init__(self, cache_dir: Path):
        self.cache = Cache(str(cache_dir))

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache.set(key, value, expire=ttl)

    def delete(self, key: str):
        self.cache.delete(key)

    def clear(self):
        self.cache.clear()


class MemoryCacheBackend(CacheBackend):
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self.cache[key] = value

    def delete(self, key: str):
        self.cache.pop(key, None)

    def clear(self):
        self.cache.clear()


class CacheManager:
    def __init__(self, config):
        self.config = config
        self.enabled = config.ENABLE_CACHE

        if not self.enabled:
            self.backend = NullCacheBackend()
            logging.info("Cache disabled")
            return

        # Initialize backend
        try:
            if config.CACHE_TYPE == "disk":
                self.backend = DiskCacheBackend(config.CACHE_DIR)
                logging.info(f"Cache initialized: disk at {config.CACHE_DIR}")
            elif config.CACHE_TYPE == "memory":
                self.backend = MemoryCacheBackend(ttl=config.CACHE_TTL)
                logging.info(f"Cache initialized: memory")
            else:
                logging.warning(
                    f"Unknown cache type: {config.CACHE_TYPE}, using NullCache"
                )
                self.backend = NullCacheBackend()
        except Exception as e:
            logging.error(f"Failed to initialize cache: {e}")
            logging.info("Using NullCache as fallback")
            self.backend = NullCacheBackend()
            self.enabled = False

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, prefix: str, *args, **kwargs) -> Optional[Any]:
        if not self.enabled:
            return None
        try:
            key = self._generate_key(prefix, *args, **kwargs)
            return self.backend.get(key)
        except Exception as e:
            logging.warning(f"Cache get failed: {e}")
            return None

    def set(self, prefix: str, value: Any, *args, **kwargs):
        if not self.enabled:
            return
        try:
            key = self._generate_key(prefix, *args, **kwargs)
            self.backend.set(key, value, ttl=self.config.CACHE_TTL)
        except Exception as e:
            logging.warning(f"Cache set failed: {e}")

    def clear_all(self):
        if self.enabled:
            try:
                self.backend.clear()
            except Exception as e:
                logging.warning(f"Cache clear failed: {e}")
