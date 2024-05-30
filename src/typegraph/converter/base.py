import inspect
import asyncio
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
    get_type_hints,
    Generic,
)
from functools import wraps, reduce

import networkx as nx
from typing_extensions import get_args, get_origin
from typing_inspect import is_union_type, get_generic_type
from typeguard import check_type, TypeCheckError, CollectionCheckStrategy

from .typevar import iter_deep_type, gen_typevar_model, extract_typevar_mapping
from ..type_utils import (
    is_structural_type,
    deep_type,
    is_protocol_type,
    check_protocol_type,
    get_subclass_types,
)


T = TypeVar("T")
In = TypeVar("In", contravariant=True)
Out = TypeVar("Out")
P = ParamSpec("P")


class TypeConverter:
    instances: List["TypeConverter"] = []

    def __init__(self):
        self.G = nx.DiGraph()
        self.sG = nx.DiGraph()
        self.pG = nx.DiGraph()
        self.tG = nx.DiGraph()
        self.pmG = nx.DiGraph()
        TypeConverter.instances.append(self)

    def get_graph(
        self,
        sub_class: bool = False,
        protocol: bool = False,
        combos: Optional[list[nx.DiGraph]] = None,
    ):
        base_graphs = [self.G]
        if sub_class:
            base_graphs.append(self.sG)
        if protocol:
            base_graphs.append(self.pG)

        if combos:
            base_graphs.extend(combos)

        if len(base_graphs) > 1:
            return nx.compose_all(base_graphs)
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

    def _gen_graph(self, in_type: Type[In], out_type: Type[Out]):
        tmp_G = nx.DiGraph()
        im = gen_typevar_model(in_type)
        om = gen_typevar_model(out_type)
        for u, v, c in self.tG.edges(data=True):
            um = gen_typevar_model(u)
            vm = gen_typevar_model(v)
            combos = [(um, im), (vm, im), (um, om), (vm, om)]
            for t, i in combos:
                try:
                    mapping = extract_typevar_mapping(t, i)
                    tmp_G.add_edge(
                        um.get_instance(mapping), vm.get_instance(mapping), **c
                    )
                except Exception:
                    ...
        return tmp_G

    def register_generic_converter(self, input_type: Type, out_type: Type):
        def decorator(func: Callable[P, T]):
            self.tG.add_edge(input_type, out_type, converter=func)
            return func

        return decorator

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
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class=False,
        protocol=False,
        combos: Optional[list[nx.DiGraph]] = None,
    ) -> bool:
        try:
            nx.has_path(
                self.get_graph(sub_class=sub_class, protocol=protocol, combos=combos),
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
        if is_union_type(out_type):
            for out_type in get_args(out_type):
                for path, converters in self.get_all_paths(
                    in_type, out_type, sub_class=sub_class, protocol=protocol
                ):
                    func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                    yield path, func
                for p_type in self.get_check_types_by_value(
                    input_value, sub_class=sub_class, protocol=protocol
                ):
                    for path, converters in self.get_all_paths(
                        p_type, out_type, sub_class=sub_class, protocol=protocol
                    ):
                        func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                        yield path, func
        for p_type in self.get_check_types_by_value(
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
        for p_type in self.get_check_types_by_value(
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
                for p_type in self.get_check_types_by_value(
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
        globalns: dict[str, Any] | None = None,
        localns: dict[str, Any] | None = None,
    ):
        def decorator(func: Callable[P, T]):
            sig = inspect.signature(func)
            hints = get_type_hints(
                func, include_extras=True, globalns=globalns, localns=localns
            )

            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                bound = sig.bind(*args, **kwargs)
                for name, value in bound.arguments.items():
                    if name in hints:
                        try:
                            bound.arguments[name] = self.convert(
                                value,
                                hints[name],
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
        globalns: dict[str, Any] | None = None,
        localns: dict[str, Any] | None = None,
    ):
        def decorator(func: Callable[P, Awaitable[T]]):
            sig = inspect.signature(func)
            hints = get_type_hints(
                func, include_extras=True, globalns=globalns, localns=localns
            )

            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                bound = sig.bind(*args, **kwargs)
                for name, value in bound.arguments.items():
                    if name in hints:
                        try:
                            bound.arguments[name] = await self.async_convert(
                                value,
                                hints[name],
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
        G = self.get_graph(
            sub_class=sub_class,
            protocol=protocol,
            combos=[self._gen_graph(in_type, out_type)],
        )
        try:
            nx.has_path(G, in_type, out_type)
            res = True
        except nx.NodeNotFound:
            res = False
        if res:
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
        yield from get_subclass_types(cls)

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

    def get_check_types_by_value(
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
            except Exception:
                continue
            yield edge[0]
