# pylint:disable-msg=E0611,I1101
"""
Functions grounding on third-party software.
"""

import logging
import os
from typing import Any, Tuple

# third-party
from justext.core import (
    ParagraphMaker,
    classify_paragraphs,
    revise_paragraph_classification,
)
from justext.utils import get_stoplist, get_stoplists
from lxml import html
from lxml.etree import _Element, strip_tags, tostring
from lxml.html import HtmlElement

# own
from .baseline import basic_cleaning
from .htmlprocessing import convert_tags, prune_unwanted_nodes, tree_cleaning
from .readability_lxml import Document as ReadabilityDocument  # fork
from .settings import JUSTEXT_LANGUAGES, normalize_fallback_name
from .utils import fromstring_bytes, trim
from .xml import TEI_VALID_TAGS
from .xpaths import OVERALL_DISCARD_XPATH

LOGGER = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

JT_STOPLIST = None
MINERU_EXTRACTOR = None
MINERU_IMPORT_ERROR = False

SANITIZED_XPATH = ".//aside|.//audio|.//button|.//fieldset|.//figure|.//footer|.//iframe|.//input|.//label|.//link|.//nav|.//noindex|.//noscript|.//object|.//option|.//select|.//source|.//svg|.//time"


def try_readability(htmlinput: HtmlElement) -> HtmlElement:
    """Safety net: try with the generic algorithm readability"""
    # defaults: min_text_length=25, retry_length=250
    try:
        doc = ReadabilityDocument(htmlinput, min_text_length=25, retry_length=250)
        # force conversion to utf-8 (see #319)
        summary = fromstring_bytes(doc.summary())
        return summary if summary is not None else HtmlElement()
    except Exception as err:
        LOGGER.warning("readability_lxml failed: %s", err)
        return HtmlElement()


def compare_extraction(
    tree: HtmlElement,
    backup_tree: HtmlElement,
    body: HtmlElement,
    text: str,
    len_text: int,
    options: Any,
) -> Tuple[HtmlElement, str, int]:
    """Decide whether to choose own or external extraction based on a series of heuristics"""
    # bypass for recall
    if options.focus == "recall" and len_text > options.min_extracted_size * 10:
        return body, text, len_text

    # Check text density
    text_density = _calculate_text_density(text, tree)
    if text_density > 0.02:
        LOGGER.debug("High text density (%s), skipping fallbacks", text_density)
        return body, text, len_text

    sanitize_output = False
    # prior cleaning
    if options.focus == "precision":
        backup_tree = prune_unwanted_nodes(backup_tree, OVERALL_DISCARD_XPATH)

    used_fallback = None

    for extractor in options.fallback_chain:
        extractor = normalize_fallback_name(extractor)
        LOGGER.debug("attempting fallback extractor: %s", extractor)

        if extractor == "readability":
            temppost_algo, algo_text, len_algo = readability_rescue(backup_tree)
            LOGGER.debug("Comparing to fallback 'readability'. Fallback length: %s (original: %s)", len_algo, len_text)
            if _should_use_html_fallback(body, len_text, temppost_algo, algo_text, len_algo, options):
                used_fallback = "readability"
                body, text, len_text = temppost_algo, algo_text, len_algo
                sanitize_output = True
                LOGGER.debug("using readability fallback: %s", options.source)
                break

        elif extractor == "justext":
            LOGGER.debug("Considering fallback 'justext'")
            if _should_try_plaintext_rescue(body, len_text, options):
                LOGGER.debug("unclean document triggering justext examination: %s",options.source,)
                body2, text2, len_text2 = justext_rescue(tree, options)
                if _is_acceptable_plaintext_rescue(len_text, text2, len_text2):
                    used_fallback = "justext"
                    LOGGER.debug("Accepting justext fallback")
                    body, text, len_text = body2, text2, len_text2
                    sanitize_output = False

        elif extractor == "mineru":
            # Pass the original tree to mineru to give it the best chance
            temppost_algo, algo_text, len_algo = mineru_rescue(tree, options)
            LOGGER.debug("Comparing to fallback 'mineru'. Fallback length: %s (original: %s)", len_algo, len_text)
            if _should_use_html_fallback(body, len_text, temppost_algo, algo_text, len_algo, options):
                used_fallback = "mineru"
                body, text, len_text = temppost_algo, algo_text, len_algo
                sanitize_output = True
                LOGGER.debug("using mineru fallback: %s", options.source)
                break

        else:
            LOGGER.warning("unknown fallback extractor skipped: %s", extractor)

    # post-processing: remove unwanted sections
    if sanitize_output:
        body, text, len_text = sanitize_tree(body, options)

    if used_fallback is not None:
        algo_text_density = _calculate_text_density(algo_text, tree)
        LOGGER.debug("Fallback extractor used: %s (text density: %s)", used_fallback, algo_text_density)

    return body, text, len_text


def _calculate_text_density(text: str, tree: HtmlElement) -> float:
    "Calculate text density as a heuristic for content quality."
    html_string = html.tostring(tree, encoding="unicode")
    return round(len(text) / len(html_string) if len(html_string) > 0 else 0.0, 3)


def _should_use_html_fallback(
    body: _Element,
    len_text: int,
    candidate_body: HtmlElement,
    candidate_text: str,
    candidate_len: int,
    options: Any,
) -> bool:
    "Use the standard external extraction heuristics for HTML candidates."
    if candidate_len in (0, len_text):
        # Reject if output is empty or same length as original
        LOGGER.debug("Rejecting fallback: empty or same length as original")
        return False
    if len_text == 0 and candidate_len > 0:
        # Accept if original is empty and candidate is not
        LOGGER.debug("Accepting fallback: original empty and candidate not empty")
        return True
    if len_text > 2 * candidate_len:
        # Reject if original is much longer than candidate
        LOGGER.debug("Rejecting fallback: original much longer than candidate")
        return False
    # quick fix for https://github.com/adbar/trafilatura/issues/632
    if candidate_len > 2 * len_text and not candidate_text.startswith("{"):
        # Accept if candidate is much longer than original and doesn't look like JSON
        LOGGER.debug("Accepting fallback: candidate much longer than original and not JSON")
        return True
    if not body.xpath(".//p//text()") and candidate_len > options.min_extracted_size * 2:
        # Accept if original has no paragraphs and candidate is reasonably long
        LOGGER.debug("Accepting fallback: original has no paragraphs and candidate is reasonably long")
        return True
    if (
        len(body.findall(".//table")) > len(body.findall(".//p"))
        and candidate_len > options.min_extracted_size * 2
    ):
        # Accept if there are more tables than paragraphs and candidate is reasonably long
        LOGGER.debug("Accepting fallback: more tables than paragraphs and candidate is reasonably long")
        return True
    # https://github.com/adbar/trafilatura/issues/354
    if (
        options.focus == "recall"
        and not body.xpath(".//head")
        and candidate_body.xpath(".//h2|.//h3|.//h4")
        and candidate_len > len_text
    ):
        # Accept if focus is recall, original has no headings, candidate has subheadings, and candidate is longer
        LOGGER.debug("Accepting fallback: focus is recall, original has no headings, candidate has subheadings, and candidate is longer")
        return True
    LOGGER.debug(
        "Rejecting fallback: no heuristics passed (extraction values: %s %s for %s)",
        len_text,
        candidate_len,
        options.source,
    )
    return False


def _should_try_plaintext_rescue(body: _Element, len_text: int, options: Any) -> bool:
    "Only run plain-text rescue methods when current output is short or noisy."
    should_try = bool(body.xpath(SANITIZED_XPATH) or len_text < options.min_extracted_size)
    if should_try:
        LOGGER.debug("considering plaintext rescue: document is short or contains unwanted sections")
    else:
        LOGGER.debug("skipping plaintext rescue: document is sufficiently long and clean")
    return should_try


def _is_acceptable_plaintext_rescue(
    len_text: int, candidate_text: str, candidate_len: int
) -> bool:
    "Prevent too short plain-text rescues from replacing the main text."
    is_acceptable = bool(candidate_text) and not len_text > 4 * candidate_len
    if is_acceptable:
        LOGGER.debug("accepting plaintext rescue: candidate is reasonably long")
    else:
        LOGGER.debug("rejecting plaintext rescue: candidate is too short")
    return is_acceptable


def readability_rescue(tree: HtmlElement) -> Tuple[HtmlElement, str, int]:
    "Extract content using readability."
    temppost_algo = try_readability(tree)
    algo_text = trim(
        tostring(temppost_algo, method="text", encoding="utf-8").decode("utf-8")
    )
    return temppost_algo, algo_text, len(algo_text)


def _get_mineru_extractor() -> Any:
    "Lazy-load the MinerU HTML extractor only when it is requested."
    global MINERU_EXTRACTOR, MINERU_IMPORT_ERROR
    if MINERU_EXTRACTOR is not None:
        return MINERU_EXTRACTOR
    if MINERU_IMPORT_ERROR:
        return None

    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
        from mineru_html import MinerUHTML_Transformers, MinerUHTMLConfig

        # Get model path from environment variable, if set
        model_path = os.getenv("MINERU_HTML_MODEL_PATH")
        if model_path is not None:
            model_path = model_path.strip() or None
            LOGGER.info("Using MinerU HTML model from environment variable: %s", model_path)

        MINERU_EXTRACTOR = MinerUHTML_Transformers(
            model_path=model_path,
            config=MinerUHTMLConfig(
                early_load=False,
                output_format="none",
                use_fall_back="empty",
            )
        )

    except Exception as err:
        MINERU_IMPORT_ERROR = True
        LOGGER.warning("mineru_html unavailable: %s", err)
        return None
    return MINERU_EXTRACTOR


def _html_to_body(htmlstring: str) -> HtmlElement:
    "Parse extracted HTML into a body element."
    result = fromstring_bytes(htmlstring)
    if result is None:
        return HtmlElement()
    if result.tag == "body":
        return result
    if result.tag == "html":
        body_elem = result.find(".//body")
        return body_elem if body_elem is not None else HtmlElement()
    body = HtmlElement("body")
    body.append(result)
    return body


def mineru_rescue(tree: HtmlElement, options: Any) -> Tuple[HtmlElement, str, int]:
    "Try to use MinerU HTML as an optional fallback extractor."
    extractor = _get_mineru_extractor()
    if extractor is None:
        return HtmlElement(), "", 0
    try:
        result = extractor.process(tostring(tree, encoding="unicode"))
        first = result[0]
        output_data = getattr(first, "output_data", None)
        main_html = getattr(output_data, "main_html", None) or getattr(first, "main_html", None)
        main_html = str(main_html).strip() if main_html else ""
    except Exception as err:
        LOGGER.warning("mineru_html failed: %s %s", err, options.url)
        return HtmlElement(), "", 0
    if not main_html:
        return HtmlElement(), "", 0
    temppost_algo = _html_to_body(main_html)
    algo_text = trim(
        tostring(temppost_algo, method="text", encoding="utf-8").decode("utf-8")
    )
    return temppost_algo, algo_text, len(algo_text)


def jt_stoplist_init() -> Tuple[str]:
    "Retrieve and return the content of all JusText stoplists"
    global JT_STOPLIST
    stoplist = set()
    for language in get_stoplists():
        stoplist.update(get_stoplist(language))
    JT_STOPLIST = tuple(stoplist)
    return JT_STOPLIST


def custom_justext(tree: HtmlElement, stoplist: Tuple[str]) -> Any:
    "Customized version of JusText processing"
    paragraphs = ParagraphMaker.make_paragraphs(tree)
    classify_paragraphs(paragraphs, stoplist, 50, 150, 0.1, 0.2, 0.25, True)
    revise_paragraph_classification(paragraphs, 150)
    return paragraphs


def try_justext(tree: HtmlElement, url: str, target_language: str) -> HtmlElement:
    """Second safety net: try with the generic algorithm justext"""
    # init
    result_body = HtmlElement("body")
    # determine language
    if target_language in JUSTEXT_LANGUAGES:
        justext_stoplist = get_stoplist(JUSTEXT_LANGUAGES[target_language])
    else:
        justext_stoplist = JT_STOPLIST or jt_stoplist_init()
    # extract
    try:
        paragraphs = custom_justext(tree, justext_stoplist)
    except Exception as err:
        LOGGER.error("justext %s %s", err, url)
    else:
        for paragraph in paragraphs:
            if paragraph.is_boilerplate:
                continue
            # if duplicate_test(paragraph) is not True:
            elem, elem.text = HtmlElement("p"), paragraph.text
            result_body.append(elem)
    return result_body


def justext_rescue(tree: HtmlElement, options: Any) -> Tuple[HtmlElement, str, int]:
    """Try to use justext algorithm as a second fallback"""
    # additional cleaning
    tree = basic_cleaning(tree)
    # proceed
    temppost_algo = try_justext(tree, options.url, options.lang)
    temp_text = trim(" ".join(temppost_algo.itertext()))
    return temppost_algo, temp_text, len(temp_text)


def sanitize_tree(tree: HtmlElement, options: Any) -> Tuple[HtmlElement, str, int]:
    """Convert and sanitize the output from the generic algorithm (post-processing)"""
    # 1. clean
    cleaned_tree = tree_cleaning(tree, options)
    if options.links is False:
        strip_tags(cleaned_tree, "a")
    strip_tags(cleaned_tree, "span")
    # 2. convert
    cleaned_tree = convert_tags(cleaned_tree, options)
    for elem in cleaned_tree.iter("td", "th", "tr"):
        # elem.text, elem.tail = trim(elem.text), trim(elem.tail)
        # finish table conversion
        if elem.tag == "tr":
            elem.tag = "row"
        elif elem.tag in ("td", "th"):
            if elem.tag == "th":
                elem.set("role", "head")
            elem.tag = "cell"
    # 3. sanitize
    sanitization_list = [
        tagname
        for tagname in [element.tag for element in set(cleaned_tree.iter("*"))]
        if tagname not in TEI_VALID_TAGS
    ]
    strip_tags(cleaned_tree, *sanitization_list)
    # 4. return
    text = trim(" ".join(cleaned_tree.itertext()))
    return cleaned_tree, text, len(text)
