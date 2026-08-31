# -*- coding: utf-8 -*-
"""
BM25 稀疏向量编码器（tcvdb_text）单例封装。

腾讯云 VectorDB 的全文/关键词检索走 sparse vector：
  - 写入：把文本编码成 sparse_vector 一并 upsert
  - 检索：fulltext_search / hybrid_search 用 query 的 sparse 表示

BM25Encoder.default('zh') 首次加载约 4s，必须进程内单例缓存。
tcvdb_text 缺失（未安装）时优雅降级：所有方法返回 None，调用方据此关闭
sparse 能力，不抛错。
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# sparse vector 类型：List[[token_id:int, weight:float], ...]
SparseVector = List[list]

_encoder = None            # BM25Encoder | None | False(=不可用)
_lock = threading.Lock()


def _get_encoder():
    """懒加载 BM25Encoder 单例。不可用时返回 False（区别于尚未初始化的 None）。"""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is not None:
            return _encoder
        try:
            from tcvdb_text.encoder.bm25 import BM25Encoder
            enc = BM25Encoder.default("zh")
            logger.info("[bm25-sparse] BM25Encoder.default('zh') loaded")
            _encoder = enc
        except Exception as e:
            logger.warning(
                f"[bm25-sparse] BM25Encoder unavailable ({e}); "
                f"sparse/fulltext disabled, hybrid degrades to dense-only"
            )
            _encoder = False
    return _encoder


def is_available() -> bool:
    """BM25 sparse 编码是否可用（tcvdb_text 已安装且加载成功）。"""
    return bool(_get_encoder())


def encode_doc(text: str) -> Optional[SparseVector]:
    """把文档文本编码为 sparse_vector；不可用或空文本返回 None。"""
    enc = _get_encoder()
    if not enc or not text:
        return None
    try:
        out = enc.encode_texts([text])  # List[SparseVector]
        if out and out[0]:
            return out[0]
        return None
    except Exception as e:
        logger.debug(f"[bm25-sparse] encode_doc failed: {e}")
        return None


def encode_docs(texts: List[str]) -> List[Optional[SparseVector]]:
    """批量编码；每条对应一个 sparse_vector 或 None（与输入等长）。"""
    enc = _get_encoder()
    if not enc:
        return [None] * len(texts)
    # 仅对非空文本编码，保持索引对齐
    idxs = [i for i, t in enumerate(texts) if t]
    result: List[Optional[SparseVector]] = [None] * len(texts)
    if not idxs:
        return result
    try:
        encoded = enc.encode_texts([texts[i] for i in idxs])
        for j, i in enumerate(idxs):
            sv = encoded[j] if j < len(encoded) else None
            result[i] = sv or None
    except Exception as e:
        logger.debug(f"[bm25-sparse] encode_docs failed: {e}")
    return result


def encode_query(text: str) -> Optional[SparseVector]:
    """把查询文本编码为 sparse_vector；不可用或空文本返回 None。"""
    enc = _get_encoder()
    if not enc or not text:
        return None
    try:
        return enc.encode_queries(text) or None
    except Exception as e:
        logger.debug(f"[bm25-sparse] encode_query failed: {e}")
        return None
