from abc import ABC
from dataclasses import dataclass

from papermage import Entity, Metadata, Span
from papermage.magelib import Document, Entity, Metadata, SentencesFieldName, Span, TokensFieldName
from papermage.predictors import BasePredictor
from tqdm.auto import tqdm


@dataclass
class EntityCharSpan:
    """Represents a tagged entity on a string."""

    e_type: str
    start_char: int
    end_char: int
    metadata: dict = None


def map_char_spans_to_entity(sentence: Entity, entities: list[EntityCharSpan]) -> list[Entity]:
    """Map a list of entities onto the given sentence's spans.

    Parameters
    ----------
    sentence : The sentence onto whose spans to map the entities. These spans will be relative to
        global document positioning, and a given sentence may have more than one span.
    entities : the entities with sentence-local spans to globalize.

    Returns
    -------
    A list of Entities with globalized spans.
    """
    all_entities = []

    # compute a map of offsets from the beginning of the sentence to every position in it
    sentence_spans = sentence.spans
    assert len(sentence.text) == sum([span.end - span.start for span in sentence_spans])
    offset_to_span_map = {}
    sentence_offset = 0
    for span_index, span in enumerate(sentence_spans):
        for span_offset in range(span.start, span.end + 1):
            offset_to_span_map[sentence_offset] = (span_index, span_offset)
            sentence_offset += 1

    # using the offset map, get a list of spans for each entity.
    for entity in entities:
        start_span_index, start_span_offset = offset_to_span_map[entity.start_char]
        entity_start = start_span_offset
        end_span_index, end_span_offset = offset_to_span_map[entity.end_char]
        entity_end = end_span_offset

        if start_span_index != end_span_index:
            start_span = Span(entity_start, sentence_spans[start_span_index.end])
            end_span = Span(sentence_spans[end_span_index.start], entity_end)
            intervening_spans = [
                Span(sentence_spans[i].start, sentence_spans[i].end)
                for i in range(start_span_index + 1, end_span_index)
            ]
            spans = [start_span] + intervening_spans + [end_span]
        else:
            spans = [Span(entity_start, entity_end)]

        all_entities.append(Entity(spans=spans, metadata=Metadata(entity_type=entity.e_type)))
    return all_entities


class TokenClassificationPredictorABC(BasePredictor, ABC):
    def __init__(self, entity_to_process="reading_order_sections"):
        self.entity_to_process = entity_to_process

    @property
    def REQUIRED_DOCUMENT_FIELDS(self) -> list[str]:
        return [self.entity_to_process]

    @property
    def predictor_identifier(self) -> str:
        """MUST IMPLEMENT! Usually the name of the underlying model."""
        raise NotImplementedError

    @property
    def preferred_layer_name(self) -> str:
        return f"TAGGED_ENTITIES_{self.predictor_identifier}"

    @property
    def entity_types(self) -> list[str]:
        raise NotImplementedError

    def tag_entities_in_batch(self, batch: list[str]) -> list[list[EntityCharSpan]]:
        """Tag entities in the given list of strings, that represents a batch.

        Parameters
        ----------
        batch : The list of strings to annotate.

        Returns
        -------
        A list of EntityCharSpans per sentence that represent the tagged entities.
        """
        raise NotImplementedError()

    def generate_batches(self, doc: Document) -> list[list[tuple[Entity, str]]]:
        """Generate batches of sentences from a document. Override this for custom batching logic.

        Parameters
        ----------
        doc : The document to use to generate batches.

        Returns
        -------
        A list of batches, each of which is a pair of Entity and its associated text. This is
        required such that we can map annotations on the string back to the document, and you can
        also apply any length-invariant transformations of the input text, e.g. replacing newlines
        with spaces, or casing.
        """
        all_batches = []
        already_processed_sentences = set()
        for para_idx, paragraph in enumerate(getattr(doc, self.entity_to_process)):
            paragraph_sentences = [
                sentence
                for sentence in paragraph.sentences
                if sentence not in already_processed_sentences
            ]
            if not paragraph_sentences:
                continue
            already_processed_sentences.update(paragraph_sentences)

            batch = [
                (sentence, sentence.text.replace("\n", " ")) for sentence in (paragraph_sentences)
            ]
            all_batches.append(batch)
        return all_batches

    def _predict(self, doc: Document) -> list[Entity]:
        all_entities = []

        for batch in tqdm(self.generate_batches(doc)):
            batch_entities, batch_texts = zip(*batch)
            tagged_by_instance = self.tag_entities_in_batch(batch_texts)
            for (instance_entity, instance_text), instance_tagged in zip(batch, tagged_by_instance):
                all_entities.extend(map_char_spans_to_entity(instance_entity, instance_tagged))

        return all_entities
