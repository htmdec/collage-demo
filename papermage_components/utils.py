import itertools
from typing import Optional

from ncls import NCLS
import numpy as np

from papermage import Document, Box, Entity, Span
from papermage.utils.merge import cluster_and_merge_neighbor_spans
from papermage.visualizers import plot_entities_on_page

from papermage_components.constants import MAT_IE_TYPES


def get_spans_from_boxes(doc: Document, boxes: list[Box]):
    intersecting_tokens = doc.intersect_by_box(query=Entity(boxes=boxes), name="tokens")
    token_spans = list(itertools.chain(*(token.spans for token in intersecting_tokens)))
    clustered_token_spans = cluster_and_merge_neighbor_spans(token_spans).merged
    filtered_for_strays = [
        merged for merged in clustered_token_spans if merged.end - merged.start > 1
    ]
    return filtered_for_strays


def get_span_by_box(box, doc) -> Optional[Span]:
    overlapping_tokens = doc.intersect_by_box(Entity(boxes=[box]), "tokens")
    token_spans = []
    for token in overlapping_tokens:
        token_spans.extend(token.spans)
    if token_spans:
        return Span.create_enclosing_span(token_spans)
    else:
        return None


def get_text_in_box(box, doc):
    cell_span = get_span_by_box(box, doc)
    return doc.symbols[cell_span.start : cell_span.end] if cell_span is not None else ""


def globalize_bbox_coordinates(bbox, context_box, doc):
    page_width, page_height = doc.pages[context_box.page].images[0].pilimage.size
    bbox_left = context_box.l + (bbox[0] / page_width)
    bbox_top = context_box.t + (bbox[1] / page_height)
    bbox_width = (bbox[2] - bbox[0]) / page_width
    bbox_height = (bbox[3] - bbox[1]) / page_height
    return Box(bbox_left, bbox_top, bbox_width, bbox_height, page=context_box.page)


def globalize_box_coordinates(box: Box, context_box: Box, doc):
    page_width, page_height = doc.pages[context_box.page].images[0].pilimage.size
    bbox_left = context_box.l + (box.l * context_box.w)
    bbox_top = context_box.t + (box.t * context_box.h)
    bbox_width = box.w * context_box.w
    bbox_height = box.h * context_box.h
    return Box(bbox_left, bbox_top, bbox_width, bbox_height, page=context_box.page)


def merge_overlapping_entities(entities):
    starts = []
    ends = []
    ids = []

    for id, entity in enumerate(entities):
        for span in entity.spans:
            starts.append(span.start)
            ends.append(span.end)
            ids.append(id)

    index = NCLS(
        np.array(starts, dtype=np.int32),
        np.array(ends, dtype=np.int32),
        np.array(ids, dtype=np.int32),
    )

    merged_entities = []
    consumed_ids = set()
    for i, entity in enumerate(entities):
        if i in consumed_ids:
            continue
        match_ids = set()
        for span in entity.spans:
            match_ids.update(
                [
                    matched_id
                    for _start, _end, matched_id in index.find_overlap(span.start, span.end)
                ]
            )
        overlapping_entities = [entities[i] for i in match_ids if i not in consumed_ids]
        if len(overlapping_entities) == 1:
            merged_entities.append(entity)
        elif len(overlapping_entities) > 1:
            all_spans = list(itertools.chain(*[entity.spans for entity in overlapping_entities]))
            if entity.boxes is not None:
                all_boxes = list(
                    itertools.chain(*[entity.boxes for entity in overlapping_entities])
                )
            else:
                all_boxes = None

            merged_entities.append(
                Entity(
                    spans=[Span.create_enclosing_span(all_spans)],
                    boxes=all_boxes,
                    metadata=list(overlapping_entities)[0].metadata,
                )
            )
            consumed_ids.update(match_ids)

    return merged_entities


def annotate_entities_on_doc(entities_by_type, spacy_doc, para_offset):
    all_spans = []
    for e_type, entities in entities_by_type.items():
        if not entities:
            continue
        for entity in entities:
            e_start_char = entity.spans[0].start - para_offset
            e_end_char = entity.spans[0].end - para_offset

            span = spacy_doc.char_span(e_start_char, e_end_char, label=e_type)
            if span is not None:
                all_spans.append(span)
    spacy_doc.set_ents(all_spans)


def visualize_highlights(paragraph_entity, spacy_pipeline):
    entities_by_type = {}
    for e in getattr(paragraph_entity, "annotation_highlights", []):
        e_type = e.metadata["annotation_type"]
        if e_type not in entities_by_type:
            entities_by_type[e_type] = []
        entities_by_type[e_type].append(e)

    para_doc = spacy_pipeline(paragraph_entity.text.replace("\n", " "))
    para_offset = paragraph_entity.spans[0].start
    if entities_by_type:
        annotate_entities_on_doc(entities_by_type, para_doc, para_offset)
    return para_doc


def visualize_tagged_entities(paragraph_entity, spacy_pipeline, model_name, allowed_entity_types):
    tagged_entities = getattr(paragraph_entity, f"TAGGED_ENTITIES_{model_name}")
    entities_by_type = {
        e_type: [e for e in tagged_entities if e.metadata["entity_type"] == e_type]
        for e_type in allowed_entity_types
    }

    para_doc = spacy_pipeline(paragraph_entity.text.replace("\n", " "))
    para_offset = paragraph_entity.spans[0].start
    annotate_entities_on_doc(entities_by_type, para_doc, para_offset)
    return para_doc


def get_table_image(table_entity: Entity, doc: Document, page_image=None, expand_box_by=0.01):
    table_images = get_table_images(table_entity, doc, page_image, expand_box_by)
    if len(table_images) != 1:
        raise AssertionError("Entity has more than one box!")
    return table_images[0]


def get_table_images(table_entity: Entity, doc: Document, page_image=None, expand_box_by=0.01):
    table_images = []
    for box in table_entity.boxes:
        if page_image is None:
            page_image = doc.pages[box.page].images[0].pilimage
        page_w, page_h = page_image.size
        table_image = page_image.crop(
            (
                (box.l - expand_box_by) * page_w,
                (box.t - expand_box_by) * page_h,
                (box.l + box.w) * page_w,
                (box.t + box.h) * page_h,
            )
        )
        table_images.append(table_image)
    return table_images


def visualize_table_with_boxes(table, boxes, doc, include_tokens):
    table_box = table.boxes[0]
    table_boxes = [Box.from_json(b) for b in boxes]
    vis_entity = plot_entities_on_page(
        doc.pages[table_box.page].images[0],
        entities=[Entity(boxes=table_boxes)],
        box_width=2,
        box_color="cornflowerblue",
    )
    if include_tokens:
        vis_entity = plot_entities_on_page(
            vis_entity, entities=table.tokens, box_width=2, box_color="red"
        )
    vis_entity = get_table_image(table, doc, vis_entity.pilimage)
    return vis_entity
