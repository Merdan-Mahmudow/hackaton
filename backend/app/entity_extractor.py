"""Модуль для извлечения имен собственных и названий компаний из текста."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Doc,
    )
    NATASHA_AVAILABLE = True
except ImportError:
    NATASHA_AVAILABLE = False

try:
    from transformers import pipeline as transformers_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class EntityExtractor:
    """Извлекает имена собственные и названия компаний из текста."""

    def __init__(self):
        self._natasha_initialized = False
        self._transformer_ner = None
        self._transformer_ner_initialized = False
        self._init_natasha()
        # Трансформер загружаем только лениво (при необходимости)

    def _init_natasha(self) -> None:
        """Инициализация Natasha для NER."""
        if not NATASHA_AVAILABLE:
            return
        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(emb)
            self.ner_tagger = NewsNERTagger(emb)
            self._natasha_initialized = True
        except Exception:
            self._natasha_initialized = False

    def _init_transformer_ner(self) -> None:
        """Ленивая инициализация трансформера для NER как fallback."""
        if self._transformer_ner_initialized:
            return
        
        self._transformer_ner_initialized = True
        
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            # Используем русскую NER модель (может быть медленной при первой загрузке)
            # Модель загружается только если Natasha недоступна
            # Можно использовать другие модели: "DeepPavlov/rubert-base-cased-conversational"
            # или "ai-forever/ruBert-base"
            self._transformer_ner = transformers_pipeline(
                "ner",
                model="DeepPavlov/rubert-base-cased-conversational",
                aggregation_strategy="simple",
                device=-1,  # CPU, можно изменить на 0 для GPU
            )
        except Exception as e:
            # Если модель не загрузилась, используем паттерны
            self._transformer_ner = None

    def extract_with_natasha(self, text: str) -> Dict[str, List[str]]:
        """Извлечение сущностей с помощью Natasha."""
        if not self._natasha_initialized:
            return {"persons": [], "organizations": [], "locations": []}

        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)

        persons = []
        organizations = []
        locations = []

        for span in doc.spans:
            if span.type == "PER":  # Person
                persons.append(span.text)
            elif span.type == "ORG":  # Organization
                organizations.append(span.text)
            elif span.type == "LOC":  # Location
                locations.append(span.text)

        return {
            "persons": list(set(persons)),  # Убираем дубликаты
            "organizations": list(set(organizations)),
            "locations": list(set(locations)),
        }

    def extract_with_transformer(self, text: str) -> Dict[str, List[str]]:
        """Извлечение сущностей с помощью трансформера."""
        if self._transformer_ner is None:
            return {"persons": [], "organizations": [], "locations": []}

        try:
            results = self._transformer_ner(text)
            persons = []
            organizations = []
            locations = []

            for entity in results:
                label = entity.get("entity_group", "").upper()
                text_entity = entity.get("word", "").strip()
                if not text_entity:
                    continue

                if "PER" in label or "PERSON" in label:
                    persons.append(text_entity)
                elif "ORG" in label or "ORGANIZATION" in label:
                    organizations.append(text_entity)
                elif "LOC" in label or "LOCATION" in label:
                    locations.append(text_entity)

            return {
                "persons": list(set(persons)),
                "organizations": list(set(organizations)),
                "locations": list(set(locations)),
            }
        except Exception:
            return {"persons": [], "organizations": [], "locations": []}

    def extract_with_patterns(self, text: str) -> Dict[str, List[str]]:
        """Простое извлечение с помощью паттернов (fallback)."""
        # Паттерны для поиска имен (с заглавной буквы, после знаков препинания)
        # Это очень простой подход, но может помочь в крайнем случае
        persons = []
        organizations = []

        # Ищем слова с заглавной буквы, которые могут быть именами
        # (после точки, запятой, начала строки)
        name_pattern = r"(?:^|\. |, |\n)([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)"
        matches = re.findall(name_pattern, text)
        for match in matches:
            word = match.strip()
            # Простая эвристика: если слово не в начале предложения и не является обычным словом
            if len(word) > 2 and word not in ["Москва", "Россия", "России"]:
                persons.append(word)

        # Ищем возможные названия компаний (слова с заглавной буквы, возможно с ООО, ЗАО и т.д.)
        org_pattern = r"(?:ООО|ЗАО|ОАО|ПАО|ИП|ИНН)\s*[«""]?([А-ЯЁ][А-Яа-яё\s""»]+)"
        org_matches = re.findall(org_pattern, text)
        organizations.extend([m.strip() for m in org_matches])
        
        # Также ищем упоминания компаний в кавычках или после "компания", "фирма" и т.д.
        org_keywords_pattern = r"(?:компания|фирма|организация|сервис|магазин|банк)\s+[«""]?([А-ЯЁ][А-Яа-яё\s""»]+)"
        org_keywords_matches = re.findall(org_keywords_pattern, text, re.IGNORECASE)
        organizations.extend([m.strip() for m in org_keywords_matches])

        return {
            "persons": list(set(persons)),
            "organizations": list(set(organizations)),
            "locations": [],
        }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Извлекает имена собственные и названия компаний из текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с ключами:
            - persons: список имен людей
            - organizations: список названий организаций/компаний
            - locations: список названий мест
        """
        if not text or not text.strip():
            return {"persons": [], "organizations": [], "locations": []}

        # Пробуем сначала Natasha (лучше для русского)
        if self._natasha_initialized:
            result = self.extract_with_natasha(text)
            # Если Natasha работает, возвращаем результат (даже если ничего не найдено)
            # Это предотвращает загрузку тяжелой трансформерной модели
            return result

        # Fallback на трансформер (только если Natasha недоступна)
        # Загружаем лениво при первом использовании
        if not self._transformer_ner_initialized:
            self._init_transformer_ner()
        
        if self._transformer_ner is not None:
            result = self.extract_with_transformer(text)
            if result["persons"] or result["organizations"]:
                return result

        # Последний fallback - простые паттерны
        return self.extract_with_patterns(text)

    def extract_entities_simple(self, text: str) -> Dict[str, Optional[str]]:
        """
        Упрощенное извлечение: пытается определить "от кого" и "к кому".

        Args:
            text: Текст для анализа

        Returns:
            Словарь с ключами:
            - from_person: имя отправителя (если найдено)
            - to_organization: название организации-получателя (если найдено)
            - to_person: имя получателя (если найдено)
        """
        entities = self.extract(text)

        from_person = None
        to_organization = None
        to_person = None

        # Берем первое найденное имя как отправителя
        if entities["persons"]:
            from_person = entities["persons"][0]

        # Берем первую найденную организацию как получателя
        if entities["organizations"]:
            to_organization = entities["organizations"][0]

        # Если есть еще имена, второе может быть получателем
        if len(entities["persons"]) > 1:
            to_person = entities["persons"][1]

        return {
            "from_person": from_person,
            "to_organization": to_organization,
            "to_person": to_person,
        }


# Глобальный экземпляр для переиспользования
_entity_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Получить глобальный экземпляр EntityExtractor."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor

