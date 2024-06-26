import types
import typing

from pydantic import TypeAdapter
import networkx as nx

from typing import (
    Union,
    List,
    Callable,
    Any,
    runtime_checkable,
    Type,
    get_args,
    get_origin,
)
from typing_extensions import get_type_hints
from typing_inspect import get_generic_type


def get_real_origin(tp):
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
            typing._BaseGenericAlias,  # type: ignore
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
    if get_real_origin(tp):
        return True
    return False


def is_protocol_type(tp):
    return hasattr(tp, "_is_protocol") and tp._is_protocol


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
        res = get_generic_type(obj)
        if res in (type, typing._GenericAlias): # type: ignore
            return Type[obj]
        return res


def attribute_check(tp, etp):
    htp = get_type_hints(tp, include_extras=True)
    hetp = get_type_hints(etp, include_extras=True)
    for key in hetp:
        if key not in htp:
            return False
        if hetp[key] != htp[key]:
            return False
    return True


def method_check(tp, etp):
    dhp = tp.__dict__
    dehp = etp.__dict__
    for key in dehp:
        if key.startswith("__") or key.startswith("_"):
            continue
        if key not in dhp:
            return False
        if not attribute_check(dhp[key], dehp[key]):
            return False
    return True


def check_protocol_type(tp, expected_type, *, strict: bool = True):
    if not is_protocol_type(expected_type):
        raise TypeError(f"{expected_type} is not a protocol type")
    if strict:
        return attribute_check(tp, expected_type) and method_check(tp, expected_type)
    return issubclass(tp, runtime_checkable(expected_type))


def generate_type(generic: Type[Any], instance: List[Type[Any]]):
    if types.UnionType == generic:
        return Union[tuple(instance)]  # type: ignore
    elif Callable == generic:
        if len(instance) == 2:
            return generic[instance[0], instance[1]]
        return generic
    elif len(instance) == 0:
        return generic
    return generic[tuple(instance)]


def get_subclass_types(cls: Type):
    if hasattr(cls, "__subclasses__"):
        for subclass in cls.__subclasses__():
            yield subclass
            yield from get_subclass_types(subclass)


def get_connected_nodes(graph, node):
    if node not in graph:
        return set()

    # 获取正向连通的节点
    successors = set(nx.descendants(graph, node))

    # 获取逆向连通的节点
    predecessors = set(nx.ancestors(graph, node))

    # 合并所有连通的节点
    connected_nodes = successors | predecessors | {node}

    return connected_nodes


def get_connected_subgraph(graph, node):
    connected_nodes = get_connected_nodes(graph, node)
    subgraph = graph.subgraph(connected_nodes).copy()
    return subgraph


def iter_type_args(tp):
    args = tp.args
    if args:
        for arg in args:
            if isinstance(arg, list):
                for i in arg:
                    yield i
                    yield from iter_type_args(i)
            else:
                yield arg
                yield from iter_type_args(arg)


def like_issubclass(tp, expected_type):
    if tp == expected_type or expected_type == Any:
        return True
    try:
        if is_protocol_type(expected_type):
            return check_protocol_type(tp, expected_type)
        if issubclass(tp, expected_type):
            return True
    except TypeError:
        if get_origin(tp) == expected_type:
            return True
    return False


def like_isinstance(obj, expected_type):
    from .converter import check_typevar_model
    res = False
    try:
        t = TypeAdapter(expected_type)
        t.validate_python(obj, strict=True)
        res = True
    except Exception:
        ...
    if res:
        return res
    if get_real_origin(expected_type) == Type:
        try:
            res = check_typevar_model(Type[obj], expected_type)
            res = True
        except Exception:
            ...
    if res:
        return res
    obj_type = deep_type(obj)
    return check_typevar_model(obj_type, expected_type)
