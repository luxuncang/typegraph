import inspect
import asyncio
import types
from typing import (
    TypeVar,
    ParamSpec,
    Callable,
    List,
    Tuple,
    Dict,
    Set,
    Iterable,
    Iterator,
    cast,
    Type,
    Awaitable,
    Any,
    Optional,
    Union,
)
from functools import wraps, reduce
from dataclasses import dataclass
from collections import deque

import networkx as nx
from typing_extensions import get_args, get_origin
from typing_inspect import is_union_type, is_typevar, get_generic_type
from typeguard import check_type, TypeCheckError, CollectionCheckStrategy


from .type_utils import (
    get_origin as get_real_origin,
    is_structural_type,
    deep_type,
    is_protocol_type,
    check_protocol_type,
)


T = TypeVar("T")
In = TypeVar("In", contravariant=True)
Out = TypeVar("Out")
P = ParamSpec("P")


def generate_type(generic: Type[Any], instance: List[Type[Any]]):
    if types.UnionType == generic:
        return Union[tuple(instance)]  # type: ignore
    return generic[tuple(instance)]


class TypeConverter:
    instances: List["TypeConverter"] = []

    def __init__(self):
        self.G = nx.DiGraph()
        self.sG = nx.DiGraph()
        self.pG = nx.DiGraph()
        self.pmG = nx.DiGraph()
        TypeConverter.instances.append(self)

    def get_graph(self, sub_class: bool = False, protocol: bool = False):
        if sub_class and protocol:
            return nx.compose_all([self.G, self.sG, self.pG])
        elif sub_class:
            return nx.compose(self.G, self.sG)
        elif protocol:
            return nx.compose(self.G, self.pG)
        return self.G

    def _gen_edge(
        self, in_type: Type[In], out_type: Type[Out], converter: Callable[P, Out]
    ):
        self.G.add_edge(in_type, out_type, converter=converter, line=True)
        for sub_in_type in self.get_subclass_types(in_type):
            self.sG.add_edge(
                sub_in_type,
                out_type,
                converter=converter,
                line=False,
                metadata={"sub_class": True},
            )
        if is_protocol_type(in_type):
            self.pmG.add_node(in_type)
        if is_protocol_type(out_type):
            self.pmG.add_node(out_type)
        for p_type in self.get_protocol_types(in_type):
            self.pG.add_edge(
                in_type,
                p_type,
                converter=lambda x: x,
                line=False,
                metadata={"protocol": True},
            )
        for p_type in self.get_protocol_types(out_type):
            if out_type == str:
                print(p_type, in_type, out_type, converter)
            self.pG.add_edge(
                out_type,
                p_type,
                converter=lambda x: x,
                line=False,
                metadata={"protocol": True},
            )

    def register_converter(self, input_type: Type[In], out_type: Type[Out]):
        def decorator(func: Callable[P, T]) -> Callable[P, Out]:
            self._gen_edge(input_type, out_type, func)
            return cast(Callable[P, Out], func)

        return decorator

    def async_register_converter(self, input_type: Type[In], out_type: Type[Out]):
        def decorator(func: Callable[P, Awaitable[Out]]):
            self._gen_edge(input_type, out_type, func)
            return func

        return decorator

    def can_convert(
        self, in_type: Type[In], out_type: Type[Out], sub_class=False, protocol=False
    ) -> bool:
        try:
            nx.has_path(
                self.get_graph(sub_class=sub_class, protocol=protocol),
                in_type,
                out_type,
            )
            res = True
        except nx.NodeNotFound:
            res = False
        return res

    def get_converter(
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class=False,
        protocol=False,
        input_value: Any = None,
    ):
        """
        [X] SubClass type
        [X] Union type
        [X] Annotated type
        [X] Structural type
        [ ] Generic type
        """

        for path, converters in self.get_all_paths(
            in_type, out_type, sub_class=sub_class, protocol=protocol
        ):
            func = reduce(lambda f, g: lambda x: g(f(x)), converters)
            yield path, func
        for p_type in self.get_protocol_types_by_value(
            input_value, sub_class=sub_class, protocol=protocol
        ):
            for path, converters in self.get_all_paths(
                p_type, out_type, sub_class=sub_class, protocol=protocol
            ):
                func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                yield path, func
        if is_union_type(out_type):
            for out_type in get_args(out_type):
                for path, converters in self.get_all_paths(
                    in_type, out_type, sub_class=sub_class, protocol=protocol
                ):
                    func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                    yield path, func
                for p_type in self.get_protocol_types_by_value(
                    input_value, sub_class=sub_class, protocol=protocol
                ):
                    for path, converters in self.get_all_paths(
                        p_type, out_type, sub_class=sub_class, protocol=protocol
                    ):
                        func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                        yield path, func

    async def async_get_converter(
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class: bool = False,
        protocol: bool = False,
        input_value=None,
    ):
        def async_wrapper(converters):
            async def async_converter(input_value):
                for converter in converters:
                    if inspect.iscoroutinefunction(converter):
                        input_value = await converter(input_value)
                    else:
                        input_value = converter(input_value)
                return input_value

            return async_converter

        for path, converters in self.get_all_paths(
            in_type, out_type, sub_class=sub_class, protocol=protocol
        ):
            yield path, async_wrapper(converters)
        for p_type in self.get_protocol_types_by_value(
            input_value, sub_class=sub_class, protocol=protocol
        ):
            for path, converters in self.get_all_paths(
                p_type, out_type, sub_class=sub_class, protocol=protocol
            ):
                yield path, async_wrapper(converters)
        if is_union_type(out_type):
            for out_type in get_args(out_type):
                for path, converters in self.get_all_paths(
                    in_type, out_type, sub_class=sub_class, protocol=protocol
                ):
                    yield path, async_wrapper(converters)
                for p_type in self.get_protocol_types_by_value(
                    input_value, sub_class=sub_class
                ):
                    for path, converters in self.get_all_paths(
                        p_type, out_type, sub_class=sub_class, protocol=protocol
                    ):
                        yield path, async_wrapper(converters)

    def _apply_converters(self, input_value, converters):
        for converter in converters:
            input_value = converter(input_value)
        return input_value

    def _get_obj_type(
        self, obj, full: bool = False, depth: int = 10, max_sample: int = -1
    ):
        if full:
            return deep_type(obj)
        return get_generic_type(obj)

    def convert(
        self,
        input_value,
        out_type: Type[Out],
        sub_class: bool = False,
        protocol: bool = False,
        debug: bool = False,
    ) -> Out:
        input_type = self._get_obj_type(input_value, full=True)
        for sub_input_type in iter_deep_type(input_type):
            all_converters = self.get_converter(
                sub_input_type,  # type: ignore
                out_type,
                sub_class=sub_class,
                protocol=protocol,
                input_value=input_value,
            )
            if all_converters is not None:
                for path, converter in all_converters:
                    try:
                        if debug:
                            print(
                                f"Converting {sub_input_type} to {out_type} using {path}, {converter}"
                            )
                        return converter(input_value)
                    except Exception:
                        ...
        if is_structural_type(input_type) and is_structural_type(out_type):
            in_origin = get_origin(input_type)
            out_origin = get_origin(out_type)
            if in_origin == out_origin:
                out_args = get_args(out_type)

                def _iter_func(item):
                    return self.convert(
                        item,
                        out_args[0],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    )

                def __iter_func_dict(item):
                    k, v = item
                    return self.convert(
                        k,
                        out_args[0],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    ), self.convert(
                        v,
                        out_args[1],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    )

                if in_origin == list or out_origin == List:
                    res = list(map(_iter_func, input_value))
                elif in_origin == tuple or out_origin == Tuple:
                    res = tuple(map(_iter_func, input_value))
                elif in_origin == set or out_origin == Set:
                    res = set(map(_iter_func, input_value))
                elif out_origin in (Iterable, Iterator):
                    res = map(_iter_func, input_value)
                elif in_origin == dict or out_origin == Dict:
                    res = dict(map(__iter_func_dict, input_value.items()))
                else:
                    raise ValueError(
                        f"Unsupported structural_type {input_type} to {out_type}"
                    )
                return cast(Out, res)

        raise ValueError(f"No converter registered for {input_type} to {out_type}")

    async def async_convert(
        self,
        input_value,
        out_type: Type[Out],
        sub_class: bool = False,
        protocol: bool = False,
        debug: bool = False,
    ) -> Out:
        input_type = self._get_obj_type(input_value, full=True)
        for sub_input_type in iter_deep_type(input_type):
            all_converters = self.async_get_converter(
                sub_input_type,  # type: ignore
                out_type,
                sub_class=sub_class,
                protocol=protocol,
                input_value=input_value,
            )
            if all_converters is not None:
                async for path, converter in all_converters:
                    try:
                        if debug:
                            print(
                                f"Converting {sub_input_type} to {out_type} using {path}, {converter}"
                            )
                        return await converter(input_value)
                    except Exception:
                        ...
        if is_structural_type(input_type) and is_structural_type(out_type):
            in_origin = get_origin(input_type)
            out_origin = get_origin(out_type)
            if in_origin == out_origin:
                out_args = get_args(out_type)

                async def _iter_func(item):
                    return await self.async_convert(
                        item,
                        out_args[0],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    )

                async def __iter_func_dict(item):
                    k, v = item
                    return await self.async_convert(
                        k,
                        out_args[0],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    ), await self.async_convert(
                        v,
                        out_args[1],
                        sub_class=sub_class,
                        protocol=protocol,
                        debug=debug,
                    )

                if in_origin == list or out_origin == List:
                    res = await asyncio.gather(*map(_iter_func, input_value))
                elif in_origin == tuple or out_origin == Tuple:
                    res = tuple(await asyncio.gather(*map(_iter_func, input_value)))
                elif in_origin == set or out_origin == Set:
                    res = set(await asyncio.gather(*map(_iter_func, input_value)))
                elif out_origin in (Iterable, Iterator):
                    res = await asyncio.gather(*map(_iter_func, input_value))
                elif in_origin == dict or out_origin == Dict:
                    items = await asyncio.gather(
                        *map(__iter_func_dict, input_value.items())
                    )
                    res = dict(items)
                else:
                    raise ValueError(
                        f"Unsupported structural_type {input_type} to {out_type}"
                    )
                return cast(Out, res)
        raise ValueError(f"No converter registered for {input_type} to {out_type}")

    def auto_convert(
        self,
        sub_class: bool = False,
        protocol: bool = False,
        ignore_error: bool = False,
    ):
        def decorator(func: Callable[P, T]):
            sig = inspect.signature(func)

            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                bound = sig.bind(*args, **kwargs)
                for name, value in bound.arguments.items():
                    param = sig.parameters[name]
                    if param.annotation is not inspect.Parameter.empty:
                        try:
                            bound.arguments[name] = self.convert(
                                value,
                                param.annotation,
                                sub_class=sub_class,
                                protocol=protocol,
                            )
                        except Exception as e:
                            if ignore_error:
                                continue
                            raise e
                return func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator

    def async_auto_convert(
        self,
        sub_class: bool = False,
        protocol: bool = False,
        ignore_error: bool = False,
    ):
        def decorator(func: Callable[P, Awaitable[T]]):
            sig = inspect.signature(func)

            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                bound = sig.bind(*args, **kwargs)
                for name, value in bound.arguments.items():
                    param = sig.parameters[name]
                    if param.annotation is not inspect.Parameter.empty:
                        try:
                            bound.arguments[name] = await self.async_convert(
                                value,
                                param.annotation,
                                sub_class=sub_class,
                                protocol=protocol,
                            )
                        except Exception as e:
                            if ignore_error:
                                continue
                            raise e
                return await func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator

    def get_edges(self, sub_class: bool = False, protocol: bool = False):
        for edge in self.get_graph(sub_class=sub_class, protocol=protocol).edges(
            data=True
        ):
            yield edge

    def show_mermaid_graph(self, sub_class: bool = False, protocol: bool = False):
        from IPython.display import display, Markdown

        text = "```mermaid\ngraph TD;\n"
        for edge in self.get_edges(sub_class=sub_class, protocol=protocol):
            line_style = "--" if edge[2]["line"] else "-.-"
            text += f"{edge[0].__name__}{line_style}>{edge[1].__name__}\n"
        text += "```"

        display(Markdown(text))
        return text

    def get_all_paths(
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class: bool = False,
        protocol: bool = False,
    ):
        G = self.get_graph(sub_class=sub_class, protocol=protocol)

        if self.can_convert(in_type, out_type, sub_class=sub_class, protocol=protocol):
            try:
                for path in nx.shortest_simple_paths(G, in_type, out_type):
                    converters = [
                        G.get_edge_data(path[i], path[i + 1])["converter"]
                        for i in range(len(path) - 1)
                    ]
                    if len(path) == 1 and len(converters) == 0:
                        yield path * 2, [lambda x: x]
                    else:
                        yield path, converters
            except nx.NetworkXNoPath:
                ...

    def get_subclass_types(self, cls: Type):
        if hasattr(cls, "__subclasses__"):
            for subclass in cls.__subclasses__():
                yield subclass
                yield from self.get_subclass_types(subclass)

    def get_protocol_types(self, cls: Type, strict: bool = True):
        nodes = set()
        for node in list(self.pmG.nodes()):
            if node in nodes:
                continue
            nodes.add(node)
            try:
                if not check_protocol_type(cls, node, strict=strict):
                    continue
            except TypeError:
                continue
            if cls != node:
                yield node

    def get_protocol_types_by_value(
        self, input_value, sub_class: bool = False, protocol: bool = False
    ):
        nodes = set()
        G = self.get_graph(sub_class=sub_class, protocol=protocol)

        for edge in G.edges():
            if edge[0] in nodes:
                continue
            nodes.add(edge[0])
            try:
                check_type(
                    input_value,
                    edge[0],
                    collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS,
                )
            except TypeCheckError:
                continue
            yield edge[0]


@dataclass
class TypeVarModel:
    origin: Any
    args: Optional[List["TypeVarModel" | List["TypeVarModel"]]] = None

    def to_dict(self):
        return {
            "origin": self.origin,
            "args": [
                arg.to_dict() if isinstance(arg, TypeVarModel) else arg
                for arg in self.args
            ]
            if self.args is not None
            else self.args,
        }

    def get_instance(self, instance: Optional[dict[Type[TypeVar], Type]] = None):
        generic = self.origin
        args_list = []
        if instance is None:
            instance = {}
        if self.args is None or not self.args:
            if is_typevar(generic):
                return instance.get(generic, generic)
            return generic

        for arg in self.args:
            if isinstance(arg, TypeVarModel):
                args_list.append(arg.get_instance(instance))
            elif isinstance(arg, list):
                args_list.append([a.get_instance(instance) for a in arg])
            else:
                raise ValueError("Invalid TypeVarModel")
        return generate_type(generic, args_list)

    def depth_first_traversal(self, parent=None, parent_arg_index=None, depth=1):
        if self.args:
            for i, arg in enumerate(self.args):
                if isinstance(arg, TypeVarModel):
                    yield from arg.depth_first_traversal(self, i, depth + 1)
                elif isinstance(arg, list):
                    for j, a in enumerate(arg):
                        if isinstance(a, TypeVarModel):
                            yield from a.depth_first_traversal(self, (i, j), depth + 1)
            yield self, parent, parent_arg_index, depth
        else:
            yield self, parent, parent_arg_index, depth

    def remove_deepest_level_node(self):
        max_depth = self.get_max_depth()
        deepest_nodes_info = []

        for node, parent, parent_arg_index, depth in self.depth_first_traversal():
            if depth == max_depth:
                deepest_nodes_info.append((node, parent, parent_arg_index))

        for _, parent, parent_arg_index in reversed(deepest_nodes_info):
            if isinstance(parent_arg_index, tuple):
                list_index, item_index = parent_arg_index
                if list_index < len(parent.args) and item_index < len(
                    parent.args[list_index]
                ):
                    parent.args[list_index].pop(item_index)
            else:
                if parent_arg_index < len(parent.args):
                    parent.args.pop(parent_arg_index)

    def level_order_traversal(self):
        current_level = deque([self])
        next_level = deque()
        while current_level:
            level_nodes = []
            while current_level:
                node = current_level.popleft()
                level_nodes.append(node.origin)
                if node.args:
                    for arg in node.args:
                        if isinstance(arg, TypeVarModel):
                            next_level.append(arg)
                        elif isinstance(arg, list):
                            for a in arg:
                                if isinstance(a, TypeVarModel):
                                    next_level.append(a)
            yield level_nodes
            current_level, next_level = next_level, deque()

    def get_max_depth(self):
        max_depth = 0

        def dfs(node, depth):
            nonlocal max_depth
            if isinstance(node, TypeVarModel):
                max_depth = max(max_depth, depth)
                if node.args:
                    for arg in node.args:
                        if isinstance(arg, TypeVarModel):
                            dfs(arg, depth + 1)
                        elif isinstance(arg, list):
                            for a in arg:
                                if isinstance(a, TypeVarModel):
                                    dfs(a, depth + 1)

        dfs(self, 1)
        return max_depth

    def __str__(self):
        return f"{self.__class__.__name__}({self.get_instance()})"


def gen_typevar_model(invar: Type[In]):
    origin = get_real_origin(invar)
    args = get_args(invar)
    if origin is None and args == ():
        return TypeVarModel(origin=invar)
    args_list = []
    for arg in args:
        if isinstance(arg, list):
            args_list.append([gen_typevar_model(a) for a in arg])
        else:
            args_list.append(gen_typevar_model(arg))
    obj = TypeVarModel(origin=origin, args=args_list)
    return obj


def iter_deep_type(invar: Type[In]):
    obj = gen_typevar_model(invar)
    max_depth = obj.get_max_depth()
    yield obj.get_instance()
    for _ in reversed(range(1, max_depth)):
        obj.remove_deepest_level_node()
        yield obj.get_instance()


def infer_generic_type(type_var: Type[Any], instance: dict[Type[TypeVar], Type]):
    model = gen_typevar_model(type_var)
    return model.get_instance(instance)
