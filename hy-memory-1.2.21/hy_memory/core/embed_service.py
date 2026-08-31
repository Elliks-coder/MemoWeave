"""
Agent Memory - EmbedService 向量化服务

将文本转换为向量表示。

功能：
- 文本向量化
- 批量向量化
- 多模型支持
- 缓存机制
- 写路径攒批队列（embed_queued）：将高并发的单条 embed 请求攒成 batch 后一次发送，
  避免大量并发 write 请求同时打爆 embedding 服务。search 路径用 embed() 不走队列。

示例：
    embed_service = EmbedService(config)
    
    # 单条向量化（search 路径，直接并发）
    vector = await embed_service.embed("用户喜欢川菜")
    
    # 写路径攒批向量化（write 路径，通过队列攒批后发送）
    vector = await embed_service.embed_queued("用户喜欢川菜")
    
    # 批量向量化
    vectors = await embed_service.embed_batch([
        "用户喜欢川菜",
        "用户住在北京",
    ])
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import hashlib
import logging

from ..config import MemoryConfig, EmbedderConfig

logger = logging.getLogger(__name__)


def _openai_version() -> str:
    """openai SDK version string (best-effort, for debug logs)."""
    try:
        import openai
        return getattr(openai, "__version__", "unknown")
    except Exception:
        return "unknown"


@dataclass
class EmbedServiceConfig:
    """
    向量化服务配置
    """
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    base_url: str = ""
    embedding_dims: int = 1536
    batch_size: int = 100
    enable_cache: bool = True
    cache_size: int = 10000
    max_retries: int = 5
    retry_delay: float = 3.0
    timeout: int = 60
    # 写路径攒批队列参数
    queue_batch_size: int = 32            # 攒满多少条立即发送
    queue_batch_window_ms: float = 1000.0  # 最多等待多少毫秒后强制发送（高并发下攒批更高效）
    extra_headers: Optional[Dict[str, str]] = None
    extra_body: Optional[Dict[str, Any]] = None


class EmbedService:
    """
    向量化服务
    
    提供文本到向量的转换能力。
    - embed()：直接发送，适合 search 路径（低延迟优先）
    - embed_queued()：通过攒批队列发送，适合 write 路径（减少并发数，避免打爆服务）
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig.from_env()
        self._embed_config = self._build_config()
        
        # 缓存
        self._cache: Dict[str, List[float]] = {}
        
        # 统计
        self._embed_count = 0
        self._cache_hits = 0

        # 写路径攒批队列（懒初始化，首次调用 embed_queued 时创建）
        self._queue_lock: Optional[asyncio.Lock] = None
        self._queue_pending: List[tuple] = []   # (text, Future)
        self._queue_flush_task: Optional[asyncio.Task] = None

        # Last embed response metadata (headers + body fields, for Venus traceId)
        self._last_embed_meta: Dict[str, Any] = {}
        
        logger.info(f"EmbedService initialized, model={self._embed_config.model}")
    
    def _build_config(self) -> EmbedServiceConfig:
        """从配置构建向量化配置"""
        embedder = self.config.embedder
        import os

        # qwen3-embedding 系列通过一站式在线服务时不支持 dimensions 参数，传了返回空
        _model = (embedder.model or "").lower()
        _base_url = (embedder.base_url or "").lower()
        dims = embedder.embedding_dims
        if ("qwen3-embedding" in _model
                and "stream-server-online-sbs-11582" in _base_url):
            dims = 0
            logger.info(f"EmbedService: cleared embedding_dims for model '{embedder.model}' via online platform")

        return EmbedServiceConfig(
            provider=embedder.provider,
            model=embedder.model,
            api_key=embedder.api_key,
            base_url=embedder.base_url,
            embedding_dims=dims,
            max_retries=embedder.max_retries if embedder.max_retries is not None else 5,
            retry_delay=embedder.retry_delay if embedder.retry_delay is not None else 3.0,
            timeout=embedder.timeout if embedder.timeout is not None else 60,
            queue_batch_size=int(os.getenv("MEMORY_EMBED_BATCH_SIZE", "32")),
            queue_batch_window_ms=float(os.getenv("MEMORY_EMBED_BATCH_WINDOW_MS", "1000")),
            extra_headers=embedder.extra_headers,
            extra_body=embedder.extra_body,
        )
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    # ----------------------------------------------------------------
    # 写路径攒批队列
    # ----------------------------------------------------------------

    async def embed_queued(self, text: str, use_cache: bool = True) -> List[float]:
        """
        写路径向量化：通过攒批队列将多个并发请求合并成一次 batch 调用。
        适合 write 路径（高并发写入时避免大量并发 embed 请求）。
        search 路径请用 embed()。
        """
        # 命中缓存直接返回，不进队列
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]

        # 懒初始化 lock（必须在 event loop 中创建）
        if self._queue_lock is None:
            self._queue_lock = asyncio.Lock()

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        async with self._queue_lock:
            self._queue_pending.append((text, fut))
            pending_count = len(self._queue_pending)

            # 攒满立即 flush
            if pending_count >= self._embed_config.queue_batch_size:
                await self._flush_queue()
            # 否则启动定时 flush 任务（若未在运行）
            elif self._queue_flush_task is None or self._queue_flush_task.done():
                self._queue_flush_task = asyncio.ensure_future(
                    self._delayed_flush(self._embed_config.queue_batch_window_ms / 1000.0)
                )

        vector = await fut

        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if len(self._cache) < self._embed_config.cache_size:
                self._cache[cache_key] = vector

        self._embed_count += 1
        return vector

    async def _delayed_flush(self, delay: float) -> None:
        """等待 delay 秒后 flush，用于定时触发"""
        await asyncio.sleep(delay)
        if self._queue_lock is not None:
            async with self._queue_lock:
                if self._queue_pending:
                    await self._flush_queue()

    async def _flush_queue(self) -> None:
        """将当前 pending 队列打包成一次 batch 请求（必须在 _queue_lock 内调用）"""
        if not self._queue_pending:
            return

        batch = self._queue_pending[:]
        self._queue_pending.clear()

        texts = [item[0] for item in batch]
        futures = [item[1] for item in batch]

        logger.debug(f"[embed_queued] flushing batch of {len(texts)}")
        try:
            vectors = await self._embed_texts(texts)
            for fut, vec in zip(futures, vectors):
                if not fut.done():
                    fut.set_result(vec)
        except Exception as e:
            logger.error(f"[embed_queued] batch failed: {e}")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
    
    async def embed(
        self,
        text: str,
        use_cache: bool = True,
    ) -> List[float]:
        """
        向量化单条文本
        
        Args:
            text: 文本内容
            use_cache: 是否使用缓存
        
        Returns:
            向量列表
        """
        # 检查缓存
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                logger.debug(f"[embed] cache hit (total hits={self._cache_hits})")
                return self._cache[cache_key]

        # 调用向量化
        vectors = await self._embed_texts([text])
        vector = vectors[0]
        
        # 更新缓存
        if use_cache and self._embed_config.enable_cache:
            if len(self._cache) < self._embed_config.cache_size:
                self._cache[cache_key] = vector
        
        self._embed_count += 1
        return vector
    
    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
    ) -> List[List[float]]:
        """
        批量向量化
        
        Args:
            texts: 文本列表
            use_cache: 是否使用缓存
        
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        results = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if use_cache and self._embed_config.enable_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    self._cache_hits += 1
                    continue
            
            texts_to_embed.append(text)
            indices_to_embed.append(i)
        
        # 批量向量化未缓存的文本
        if texts_to_embed:
            vectors = await self._embed_texts(texts_to_embed)
            
            for j, vector in enumerate(vectors):
                idx = indices_to_embed[j]
                results[idx] = vector
                
                # 更新缓存
                if use_cache and self._embed_config.enable_cache:
                    cache_key = self._get_cache_key(texts_to_embed[j])
                    if len(self._cache) < self._embed_config.cache_size:
                        self._cache[cache_key] = vector
            
            self._embed_count += len(texts_to_embed)
        
        return results
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """实际的向量化调用（统一走 OpenAI 兼容协议），逐条调用避免 batch size 限制"""
        all_vectors: List[List[float]] = []
        for text in texts:
            vectors = await self._embed_openai([text])
            all_vectors.extend(vectors)
        return all_vectors
    
    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """
        使用 OpenAI 兼容接口向量化

        支持所有 OpenAI 兼容的 embedding 服务：
        - OpenAI 官方
        - 阿里云 DashScope (text-embedding-v3)
        - DeepSeek
        - 本地 vLLM / Ollama 等
        - 内网 qwen3-embedding (turbotke openai_infer 协议)

        请求参数：model + input (+ dimensions) + extra_headers + extra_body。
        dimensions 由 _build_config 的 online 平台 guard 置 0 时不带上。

        成功后会将 response headers 和 body metadata 存入 self._last_embed_meta，
        供 embed_with_meta() 透出给 trace log（用于与 Venus 对账 traceId）。
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

        # 仅对以下"疑似网络/服务端抖动"的错误做重试；业务错（400/401 等）立即抛。
        # 使用字符串匹配是为了避免额外 import openai 的具体异常类型（不同版本命名有变）。
        _RETRYABLE = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
        )

        kwargs: Dict[str, Any] = {
            "model": self._embed_config.model,
            "input": texts,
        }
        if self._embed_config.embedding_dims:
            kwargs["dimensions"] = self._embed_config.embedding_dims
        if self._embed_config.extra_headers:
            kwargs["extra_headers"] = self._embed_config.extra_headers
        if self._embed_config.extra_body:
            kwargs["extra_body"] = self._embed_config.extra_body

        _max_retries = max(0, int(self._embed_config.max_retries or 0))
        _retry_delay = max(0.0, float(self._embed_config.retry_delay or 0.0))
        last_exc: Optional[Exception] = None

        for attempt in range(_max_retries + 1):
            # 每次尝试都新建 client，避免上一次卡死的连接被复用。
            client = AsyncOpenAI(
                api_key=self._embed_config.api_key,
                base_url=self._embed_config.base_url or None,
                timeout=self._embed_config.timeout or 3,
            )
            try:
                # Use with_raw_response to capture HTTP headers (Venus traceId etc.)
                raw = await client.embeddings.with_raw_response.create(**kwargs)
                response = raw.parse()

                # Store metadata for trace logging (headers + body fields except vectors)
                _sensitive = {"authorization", "set-cookie", "cookie"}
                resp_headers = {
                    k: v for k, v in raw.headers.items()
                    if k.lower() not in _sensitive
                }
                self._last_embed_meta = {
                    "http_status": raw.status_code,
                    "response_headers": resp_headers,
                    "model": response.model,
                    "usage": response.usage.model_dump() if response.usage else None,
                    "attempts": attempt + 1,
                }

                return [item.embedding for item in response.data]
            except Exception as e:
                last_exc = e
                _err_name = type(e).__name__
                _err_msg = str(e)
                # 判断是否可重试：命中显式异常类型 或 消息里出现常见超时/连接关键字。
                _retryable = isinstance(e, _RETRYABLE) or any(
                    kw in _err_msg.lower()
                    for kw in ("timeout", "timed out", "connection", "read operation")
                )
                self._last_embed_meta = {
                    "error": _err_name,
                    "error_msg": _err_msg[:300],
                    "attempts": attempt + 1,
                    "retryable": _retryable,
                }
                if _retryable and attempt < _max_retries:
                    logger.warning(
                        f"[embed] attempt {attempt + 1}/{_max_retries + 1} failed "
                        f"({_err_name}): {_err_msg[:200]}; retry in {_retry_delay}s"
                    )
                    if _retry_delay > 0:
                        await asyncio.sleep(_retry_delay)
                    continue
                logger.warning(
                    f"[embed] failed after {attempt + 1} attempt(s) "
                    f"({_err_name}): {_err_msg[:200]}"
                )
                raise

        # 理论不可达；兜底重抛
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("embed_openai: unreachable state")

    async def embed_with_meta(
        self,
        text: str,
        use_cache: bool = True,
    ) -> tuple:
        """
        向量化单条文本，同时返回 provider metadata（含 Venus traceId 等）。

        Returns:
            (vector, meta_dict)
            meta_dict 包含 http_status, response_headers, model, usage 等。
            如果命中缓存，meta_dict 为 {"cache_hit": True}。
        """
        # Check cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key], {"cache_hit": True}

        # Call embedding (will populate self._last_embed_meta)
        self._last_embed_meta = {}
        vectors = await self._embed_texts([text])
        vector = vectors[0]

        # Update cache
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if len(self._cache) < self._embed_config.cache_size:
                self._cache[cache_key] = vector

        self._embed_count += 1
        meta = getattr(self, "_last_embed_meta", {})
        meta["cache_hit"] = False
        return vector, meta


    async def embed_with_cache_info(
        self,
        text: str,
        use_cache: bool = True,
    ) -> tuple:
        """
        向量化单条文本，返回是否命中缓存
        
        Args:
            text: 文本内容
            use_cache: 是否使用缓存
        
        Returns:
            (向量列表, 是否命中缓存)
        """
        # 检查缓存
        if use_cache and self._embed_config.enable_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key], True
        
        # 调用向量化
        vectors = await self._embed_texts([text])
        vector = vectors[0]
        
        # 更新缓存
        if use_cache and self._embed_config.enable_cache:
            if len(self._cache) < self._embed_config.cache_size:
                self._cache[cache_key] = vector
        
        self._embed_count += 1
        return vector, False
    
    def get_embedding_dims(self) -> int:
        """获取向量维度"""
        return self._embed_config.embedding_dims
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("EmbedService cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "embed_count": self._embed_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._cache_hits / (self._embed_count + self._cache_hits)
                if (self._embed_count + self._cache_hits) > 0 else 0
            ),
            "model": self._embed_config.model,
            "embedding_dims": self._embed_config.embedding_dims,
        }
