# utils/text.py

import re
from difflib import get_close_matches
from typing import Iterable, List
from w3lib.html import remove_tags


# ---------------- NLTK：懒加载，避免导入时爆资源错误 ----------------

def _ensure_nltk():
    """在需要用到分词时才检查/下载 NLTK 资源。"""
    try:
        import nltk  # 局部导入，避免模块导入即失败
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass
        # NLTK 3.8+ 拆分了 punkt_tab
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
    except Exception:
        # 极端情况下（环境受限）静默失败，调用方会用降级策略
        pass


def _safe_tokenize(text: str) -> List[str]:
    """尽量用 nltk 分词；失败则退回最简单的空白切分。"""
    try:
        _ensure_nltk()
        from nltk.tokenize import word_tokenize  # 懒导入
        return word_tokenize(text)
    except Exception:
        return text.split()


# ---------------- 关键词匹配：保留以便检索/过滤时使用 ----------------

def match_keywords(text: str, keywords: Iterable[str], method: str = "simple", fuzzy_threshold: float = 0.8) -> List[str]:
    if not text or not keywords:
        return []

    # 统一预处理
    text_lower = text.lower()
    keywords = list(keywords)

    if method == "simple":
        return [kw for kw in keywords if kw and kw.lower() in text_lower]

    elif method == "regex":
        matched = []
        for kw in keywords:
            if not kw:
                continue
            # \b 适合英文单词边界；若是短语/包含标点，直接用 re.escape(kw) 做普通搜索也可
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                matched.append(kw)
        return matched

    elif method == "token":
        tokens = set(_safe_tokenize(text_lower))
        return [kw for kw in keywords if kw and kw.lower() in tokens]

    elif method == "fuzzy":
        # 小写做匹配，但映射回原始关键词
        lower2orig = {}
        for kw in keywords:
            if not kw:
                continue
            kl = kw.lower()
            lower2orig.setdefault(kl, kw)

        tokens = set(_safe_tokenize(text_lower))
        hits = set()
        for token in tokens:
            close = get_close_matches(token, list(lower2orig.keys()), n=1, cutoff=fuzzy_threshold)
            if close:
                hits.add(lower2orig[close[0]])
        return sorted(hits)

    else:
        raise ValueError(f"Unsupported matching method: {method}")


# ---------------- 抓取辅助 ----------------

def get_text_safe(selector, default: str = "") -> str:
    """
    安全地从 Scrapy Selector 提取字符串并 strip；失败返回 default。
    """
    try:
        value = selector.get(default)
        return value.strip() if value else default
    except Exception:
        return default


def extract_clean_content(response, selectors: Iterable[str]) -> str:
    """
    从多个 CSS 选择器中提取纯文本并拼接，去掉空段落。
    """
    if not selectors:
        return ""

    content_parts: List[str] = []
    for css in selectors:
        html_parts = response.css(css).getall()
        for p in html_parts:
            p = (p or "").strip()
            if p:
                content_parts.append(remove_tags(p).strip())

    return " ".join(content_parts)


def count_words(text: str) -> int:
    """
    简单的词数统计（空白切分）；用于粗略统计。
    """
    return len(text.strip().split()) if text else 0


# ---------------- 关键词工具 ----------------

def merge_keywords(tfidf_kws: Iterable[str], textrank_kws: Iterable[str]) -> List[str]:
    """
    合并两套关键词（去重）。
    """
    tfidf_kws = tfidf_kws or []
    textrank_kws = textrank_kws or []
    return list(set(tfidf_kws) | set(textrank_kws))


def extract_textrank_keywords(text: str, topn: int = 10) -> List[str]:
    """
    可选：用 gensim 的 TextRank 提取关键词。
    - 如果 gensim 不可用或报错，返回 []，不阻塞主流程。
    """
    if not text:
        return []
    try:
        from gensim.summarization import keywords as textrank_keywords  # 懒导入，避免环境问题
        result = textrank_keywords(text, words=topn, split=True, lemmatize=True)
        # 去空、去重
        result = [kw.strip() for kw in result if kw and kw.strip()]
        # 有些版本返回已排序，这里不强制重排
        return list(dict.fromkeys(result))
    except Exception:
        return []
