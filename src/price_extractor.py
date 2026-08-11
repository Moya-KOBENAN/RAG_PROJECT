import re
import unicodedata


STOP_WORDS = {
    "au", "aux", "combien", "de", "des", "du", "est", "la", "le", "les",
    "moyen", "moyenne", "prix", "quel", "quelle", "quels", "quelles", "un",
    "une", "variation",
}

ROW_PATTERN = re.compile(
    r"^\s*\d+\s+(?P<product>.+?)\s+"
    r"(?P<quantity>(?:sac\s+de\s+)?\d+(?:[,.]\d+)?\s*(?:kg|g|l))\s+"
    r"(?P<prices>[\d\s]+?)\s+(?P<variation>[+-]?\d+(?:[,.]\d+)?%)\s*$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def _product_terms(question: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", _normalize(question))
        if len(word) >= 3 and word not in STOP_WORDS
    }


def _terms_match(terms: set[str], words: set[str]) -> bool:
    return any(
        term == word
        or (
            min(len(term), len(word)) >= 4
            and (term.startswith(word) or word.startswith(term))
        )
        for term in terms
        for word in words
    )


def _split_two_prices(raw_prices: str) -> tuple[str, str] | None:
    parts = raw_prices.split()
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3:
        return " ".join(parts[:2]), parts[2]
    if len(parts) == 4:
        return " ".join(parts[:2]), " ".join(parts[2:])
    return None


def _context(text: str) -> tuple[str | None, str | None]:
    region_match = re.search(r"\bDR\s+([^/\n]+?)\s*/", text, re.IGNORECASE)
    period_match = re.search(
        r"\b(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|"
        r"octobre|novembre|décembre)\s+\d{4}\b",
        text,
        re.IGNORECASE,
    )
    region = region_match.group(1).strip().title() if region_match else None
    period = period_match.group(0).strip() if period_match else None
    return region, period


def extract_price_answer(question: str, results: list[dict]) -> str | None:
    """Construit une réponse factuelle depuis les lignes de tableaux récupérées."""
    if "prix" not in _normalize(question) and "cout" not in _normalize(question):
        return None

    terms = _product_terms(question)
    if not terms:
        return None

    findings = []
    period = None
    for result in results:
        text = result["text"]
        region, document_period = _context(text)
        period = period or document_period

        for line in text.splitlines():
            line_words = set(re.findall(r"[a-z0-9]+", _normalize(line)))
            if not _terms_match(terms, line_words):
                continue
            match = ROW_PATTERN.match(line)
            if not match:
                continue
            prices = _split_two_prices(match.group("prices"))
            if not prices:
                continue
            _, current_price = prices
            findings.append(
                {
                    "product": match.group("product").strip(),
                    "quantity": match.group("quantity").strip(),
                    "price": current_price,
                    "region": region,
                }
            )
            break

    if not findings:
        return None

    product = findings[0]["product"]
    details = []
    for finding in findings:
        location = f" à {finding['region']}" if finding["region"] else ""
        details.append(
            f"{finding['price']} FCFA pour {finding['quantity']}{location}"
        )
    date = f" en {period}" if period else ""
    return f"Le prix moyen de {product}{date} est de " + " et de ".join(details) + "."


def extract_variation_answer(question: str, results: list[dict]) -> str | None:
    """Décrit les hausses et baisses de prix trouvées dans les tableaux."""
    if "variation" not in _normalize(question):
        return None

    terms = _product_terms(question)
    if not terms:
        return None

    findings = []
    period = None
    for result in results:
        text = result["text"]
        region, document_period = _context(text)
        period = period or document_period

        for line in text.splitlines():
            line_words = set(re.findall(r"[a-z0-9]+", _normalize(line)))
            if not _terms_match(terms, line_words):
                continue
            match = ROW_PATTERN.match(line)
            if not match:
                continue
            variation = match.group("variation")
            numeric_variation = float(variation.rstrip("%").replace(",", "."))
            findings.append(
                {
                    "product": match.group("product").strip(),
                    "variation": variation.lstrip("+"),
                    "numeric_variation": numeric_variation,
                    "region": region,
                }
            )
            break

    if not findings:
        return None

    details = []
    for finding in findings:
        value = finding["variation"].lstrip("-")
        if finding["numeric_variation"] < 0:
            movement = f"a diminué de {value}"
        elif finding["numeric_variation"] > 0:
            movement = f"a augmenté de {value}"
        else:
            movement = "est resté stable (0%)"
        location = f" à {finding['region']}" if finding["region"] else ""
        details.append(f"{movement}{location}")

    product = findings[0]["product"]
    date = f" en {period}" if period else ""
    return f"Le prix de {product}{date} " + " et ".join(details) + "."


def extract_general_answer(question: str, results: list[dict]) -> str | None:
    """Retourne les lignes les plus directement liées à la question."""
    terms = _product_terms(question)
    excerpts = []

    for result in results:
        source = result.get("source") or "document"
        page = result.get("page")
        location = f"{source}, page {page}" if page is not None else source

        for line in result["text"].splitlines():
            clean_line = " ".join(line.split())
            if not clean_line:
                continue
            line_words = set(re.findall(r"[a-z0-9]+", _normalize(clean_line)))
            if terms and not _terms_match(terms, line_words):
                continue
            excerpt = f"{clean_line} ({location})"
            if excerpt not in excerpts:
                excerpts.append(excerpt)
            if len(excerpts) == 3:
                break
        if len(excerpts) == 3:
            break

    if not excerpts:
        for result in results:
            fallback = " ".join(result["text"].split())
            if fallback:
                source = result.get("source") or "document"
                page = result.get("page")
                location = f"{source}, page {page}" if page is not None else source
                excerpts.append(f"{fallback[:500]} ({location})")
                break
    if not excerpts:
        return None
    return "Informations trouvées : " + " ; ".join(excerpts) + "."


def extract_answer(question: str, results: list[dict]) -> str | None:
    """Produit une réponse extractive, avec une règle précise pour les prix."""
    return (
        extract_variation_answer(question, results)
        or extract_price_answer(question, results)
        or extract_general_answer(question, results)
    )
