import types
import typing
import inspect

from typing import TypeVar, Generic, Union, Annotated, List, Dict, Callable, Any
from typing_extensions import get_type_hints
from typing_inspect import is_generic_type, get_generic_type


def get_origin(tp):
    """Get the unsubscripted version of a type.

    This supports generic types, Callable, Tuple, Union, Literal, Final, ClassVar
    and Annotated. Return None for unsupported types. Examples::

        get_origin(Literal[42]) is Literal
        get_origin(int) is None
        get_origin(ClassVar[int]) is ClassVar
        get_origin(Generic) is Generic
        get_origin(Generic[T]) is Generic
        get_origin(Union[T, int]) is Union
        get_origin(List[Tuple[T, T]][int]) == List
        get_origin(P.args) is P
    """
    if isinstance(tp, typing._AnnotatedAlias):  # type: ignore
        return typing.Annotated
    if isinstance(tp, typing._GenericAlias):  # type: ignore
        if isinstance(tp._name, str) and getattr(typing, tp._name, None):
            return getattr(typing, tp._name)
        return tp.__origin__
    if isinstance(
        tp,
        (
            typing._BaseGenericAlias,
            typing.GenericAlias,  # type: ignore
            typing.ParamSpecArgs,
            typing.ParamSpecKwargs,
        ),
    ):
        return tp.__origin__
    if tp is typing.Generic:
        return typing.Generic
    if isinstance(tp, types.UnionType):
        return types.UnionType
    return None


def is_structural_type(tp):
    if get_origin(tp):
        return True
    return False


def deep_type(obj, depth: int = 10, max_sample: int = -1):
    if depth <= 0:
        return get_generic_type(obj)
    if isinstance(obj, dict):
        keys = set()
        values = set()
        for k, v in obj.items():
            keys.add(deep_type(k, depth - 1, max_sample))
            values.add(deep_type(v, depth - 1, max_sample))
        if len(keys) == 1 and len(values) == 1:
            return dict[(*tuple(keys), *tuple(values))]  # type: ignore
        elif len(keys) > 1 and len(values) == 1:
            k_tpl = Union[tuple(keys)]  # type: ignore
            return dict[(k_tpl, *values)]  # type: ignore
        elif len(keys) == 1 and len(values) > 1:
            v_tpl = Union[tuple(values)]  # type: ignore
            return dict[(*keys, v_tpl)]  # type: ignore
        elif len(keys) > 1 and len(values) > 1:
            k_tpl = Union[tuple(keys)]  # type: ignore
            v_tpl = Union[tuple(values)]  # type: ignore
            return dict[(k_tpl, v_tpl)]  # type: ignore
        else:
            return dict
    elif isinstance(obj, list):
        args = set()
        for i in obj[::max_sample]:
            args.add(deep_type(i, depth - 1, max_sample))
        if len(args) == 1:
            return list[tuple(args)]  # type: ignore
        elif len(args) > 1:
            tpl = Union[tuple(args)]  # type: ignore
            return list[tpl]  # type: ignore
        else:
            return list
    elif isinstance(obj, tuple):
        args = []
        for i in obj:
            args.append(deep_type(i, depth - 1, max_sample))
        if len(args) >= 1:
            return tuple[tuple(args)]  # type: ignore
        else:
            return tuple
    else:
        return get_generic_type(obj)
