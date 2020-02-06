from nltk.corpus import wordnet as wn

from .constants import IMAGENET


def get_labels(class_name: str):
    class_name = class_name.replace(" ", "_")

    original_synsets = wn.synsets(class_name)
    original_synsets = list(filter(lambda s: s.pos() == 'n', original_synsets))  # keep only names
    if not original_synsets:
        return None

    result = list(filter(lambda s: s.offset() in IMAGENET, original_synsets))
    if result:
        return {ps.name().split('.')[0]: IMAGENET[ps.offset()] for ps in result}
    else:
        result = original_synsets[:]
        first = sum([s.hyponyms() for s in original_synsets], [])
        while first:
            result.extend(first)
            first = sum([s.hyponyms() for s in first], [])

        result = list(filter(lambda s: s.offset() in IMAGENET, result))
        return {ps.name().split('.')[0]: IMAGENET[ps.offset()] for ps in result}
