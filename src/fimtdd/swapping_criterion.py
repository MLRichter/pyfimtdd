from .domain import SwappingCriterion
from attr import attrs, attrib


@attrs(auto_attribs=True, frozen=True, slots=True)
class DecayingSquaredErrorSwapper:
    ...
