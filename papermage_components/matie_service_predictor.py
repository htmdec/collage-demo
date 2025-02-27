from dataclasses import asdict, dataclass
import difflib
import os
import re
import subprocess
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Union

from papermage.magelib import (
    Document,
    Entity,
    Metadata,
    Prediction,
    SentencesFieldName,
    Span,
    TokensFieldName,
)
from papermage.predictors import BasePredictor
from papermage.utils.annotate import group_by
from pydantic import TypeAdapter
import requests

from papermage_components.constants import MAT_IE_TYPES


@dataclass
class MatIEEntity:
    id: str
    entity_type: str
    start: int
    end: int
    entity_string: str

    def to_papermage_entity(self):
        span = Span(self.start, self.end)
        meta = Metadata(
            entity_type=self.entity_type, entity_string=self.entity_string, entity_id=self.id
        )
        return Entity(spans=[span], metadata=meta)


relation_re = re.compile("R\d+\t(?P<r_type>.*) Arg1:(?P<arg1>T\d+) Arg2:(?P<arg2>T\d+)")


def get_offset_map(in_text, out_text):
    matcher = difflib.SequenceMatcher(isjunk=lambda x: False, a=out_text, b=in_text, autojunk=False)
    opcodes = matcher.get_opcodes()
    offsets = {}

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = j
        elif tag == "delete":
            assert j1 == j2
            for i in range(i1, i2):
                offsets[i] = j1
        elif tag == "insert":
            # we shouldn't need to do anything here, as long as we only care about matching out to in.
            pass
        elif tag == "replace":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                offsets[i] = None
    offsets[len(out_text)] = len(in_text)
    # offsets = {v: k for k, v in offsets.items()}
    return offsets


def fix_entity_offsets(entities, offset_map, para_offset):
    updated_entities = []
    for entity in entities:
        start_offset_file = offset_map[entity.start]
        end_offset_file = offset_map[entity.end]

        updated_entities.append(
            MatIEEntity(
                entity.id,
                entity.entity_type,
                start_offset_file + para_offset,
                end_offset_file + para_offset,  # start_offset_file + len(entity.entity_string) +
                # para_offset,
                entity.entity_string,
            )
        )
    return updated_entities


@dataclass
class MatIERelation:
    id: str
    relation_type: str
    arg1: str
    arg2: str


MatIEResponse = Dict[str, Dict[str, Union[str, List[MatIEEntity], List[MatIERelation]]]]


def construct_document_payload(doc: Document) -> dict[str, str]:
    input_paragraphs = {}
    for paragraph in doc.reading_order_sections:
        section_name = paragraph.metadata["section_name"]
        paragraph_order = paragraph.metadata["paragraph_reading_order"]
        # TODO: make this more robust!! This implicitly assumes a paragraph has only one span.
        paragraph_text = paragraph.text.replace("\n", " ")
        if len(paragraph.spans) != 0:
            key = f"{section_name}_{paragraph_order}"
            input_paragraphs[key] = paragraph_text
    return input_paragraphs


class MatIEServicePredictor(BasePredictor):
    def __init__(
        self,
        matIE_service_url,
    ):
        self.service_url = matIE_service_url
        self.preferred_layer_name = "TAGGED_ENTITIES_MatIE"

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> List[str]:
        return [SentencesFieldName, TokensFieldName]

    @property
    def entity_types(self):
        return MAT_IE_TYPES

    @property
    def predictor_identifier(self):
        return "MatIE"

    def _predict(self, doc: Document) -> List[Entity]:
        document_payload = construct_document_payload(doc)

        results = requests.post(
            self.service_url + "/annotate_strings", json=document_payload, timeout=600
        )

        annotated_content = TypeAdapter(MatIEResponse).validate_python(results.json())

        fixed_entities = []
        for (key, input_text), paragraph in zip(
            document_payload.items(), doc.reading_order_sections
        ):
            para_offset = paragraph.spans[0].start
            annotated_text = annotated_content[key]["text"]
            offset_map = get_offset_map(input_text, annotated_text)
            fixed_entities.extend(
                fix_entity_offsets(annotated_content[key]["entities"], offset_map, para_offset)
            )
            paragraph.metadata["in_section_relations"] = [
                asdict(r) for r in annotated_content[key]["relations"]
            ]
        papermage_entities = [entity.to_papermage_entity() for entity in fixed_entities]
        return papermage_entities
