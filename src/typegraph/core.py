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
    Annotated,
)
from functools import wraps, reduce, partial
from dataclasses import dataclass

from typing_extensions import get_args, get_origin
from typing_inspect import is_union_type, is_typevar, get_generic_type
import networkx as nx

from .type_utils import get_origin as get_real_origin, is_structural_type, deep_type


T = TypeVar("T")
In = TypeVar("In", contravariant=True)
Out = TypeVar("Out")
P = ParamSpec("P")


def get_all_subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from get_all_subclasses(subclass)


def generate_type(generic: Type[Any], instance: List[Type[Any]]):
    return generic[tuple(instance)]


def _wrapper_converter(origin, func, datas):
    return reduce(func)  # type: ignore


class TypeConverter:
    instances: List["TypeConverter"] = []

    def __init__(self):
        self.G = nx.DiGraph()
        self.vG = nx.DiGraph()
        self.gG = nx.DiGraph()
        TypeConverter.instances.append(self)

    def _gen_edge(
        self, in_type: Type[In], out_type: Type[Out], converter: Callable[P, Out]
    ):
        self.G.add_edge(in_type, out_type, converter=converter, line=True)
        self.vG.add_edge(in_type, out_type, converter=converter, line=True)
        for sub_in_type in get_all_subclasses(in_type):
            self.vG.add_edge(sub_in_type, out_type, converter=converter, line=False)

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
        self, in_type: Type[In], out_type: Type[Out], full: bool = False
    ) -> bool:
        try:
            if full:
                nx.has_path(self.vG, in_type, out_type)
            else:
                nx.has_path(self.G, in_type, out_type)
            res = True
        except nx.NodeNotFound:
            res = False
        return res

    def get_converter(
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class: bool = False,
        input_value: Any = None,
        origin_func: Optional[Callable] = None,
    ):
        """
        [X] SubClass type
        [X] Union type
        [X] Annotated type
        [X] Structural type
        [ ] Generic type
        """

        if self.can_convert(in_type, out_type, full=sub_class):
            for path, converters in self.get_all_paths(
                in_type, out_type, full=sub_class
            ):
                func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                if origin_func:
                    yield path, partial(partial(_wrapper_converter, origin_func), func)
                else:
                    yield path, func
        if is_union_type(out_type):
            for out_type in get_args(out_type):
                if self.can_convert(in_type, out_type, full=sub_class):
                    for path, converters in self.get_all_paths(
                        in_type, out_type, full=sub_class
                    ):
                        func = reduce(lambda f, g: lambda x: g(f(x)), converters)
                        if origin_func:
                            yield (
                                path,
                                partial(partial(_wrapper_converter, origin_func), func),
                            )
                        else:
                            yield path, func

    async def async_get_converter(
        self,
        in_type: Type[In],
        out_type: Type[Out],
        sub_class: bool = False,
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

        if self.can_convert(in_type, out_type, full=sub_class):
            for path, converters in self.get_all_paths(
                in_type, out_type, full=sub_class
            ):
                yield path, async_wrapper(converters)
        if is_union_type(out_type):
            for out_type in get_args(out_type):
                if self.can_convert(in_type, out_type, full=sub_class):
                    for path, converters in self.get_all_paths(
                        in_type, out_type, full=sub_class
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
        debug: bool = False,
    ) -> Out:
        input_type = self._get_obj_type(input_value, full=True)
        all_converters = self.get_converter(
            input_type, out_type, sub_class, input_value
        )
        if all_converters is not None:
            for path, converter in all_converters:
                try:
                    if debug:
                        print(
                            f"Converting {input_type} to {out_type} using {path}, {converter}"
                        )
                    return converter(input_value)
                except Exception:
                    continue
        if is_structural_type(input_type) and is_structural_type(out_type):
            in_origin = get_origin(input_type)
            out_origin = get_origin(out_type)
            if in_origin == out_origin:
                in_args = get_args(input_type)
                out_args = get_args(out_type)

                def _iter_func(item):
                    return self.convert(
                        item, out_args[0], sub_class=sub_class, debug=debug
                    )

                def __iter_func_dict(item):
                    k, v = item
                    return self.convert(
                        k, out_args[0], sub_class=sub_class, debug=debug
                    ), self.convert(v, out_args[1], sub_class=sub_class, debug=debug)

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
        debug: bool = False,
    ) -> Out:
        input_type = self._get_obj_type(input_value, full=True)
        all_converters = self.async_get_converter(
            input_type, out_type, sub_class, input_value
        )
        if all_converters is not None:
            async for path, converter in all_converters:
                try:
                    if debug:
                        print(
                            f"Converting {input_type} to {out_type} using {path}, {converter}"
                        )
                    return await converter(input_value)
                except Exception:
                    continue
        if is_structural_type(input_type) and is_structural_type(out_type):
            in_origin = get_origin(input_type)
            out_origin = get_origin(out_type)
            if in_origin == out_origin:
                in_args = get_args(input_type)
                out_args = get_args(out_type)

                async def _iter_func(item):
                    return await self.async_convert(
                        item, out_args[0], sub_class=sub_class, debug=debug
                    )

                async def __iter_func_dict(item):
                    k, v = item
                    return await self.async_convert(
                        k, out_args[0], sub_class=sub_class, debug=debug
                    ), await self.async_convert(
                        v, out_args[1], sub_class=sub_class, debug=debug
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

    def auto_convert(self, sub_class: bool = False):
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
                                value, param.annotation, sub_class=sub_class
                            )
                        except ValueError:
                            continue
                return func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator

    def async_auto_convert(self, sub_class: bool = False):
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
                                value, param.annotation, sub_class=sub_class
                            )
                        except ValueError:
                            continue
                return await func(*bound.args, **bound.kwargs)

            return wrapper

        return decorator

    def get_edges(self, full: bool = False):
        if not full:
            for edge in self.G.edges(data=True):
                yield edge
        else:
            for edge in self.vG.edges(data=True):
                yield edge

    def show_mermaid_graph(self, full: bool = False):
        from IPython.display import display, Markdown

        text = "```mermaid\ngraph TD;\n"
        for edge in self.get_edges(full=full):
            line_style = "--" if edge[2]["line"] else "-.-"
            text += f"{edge[0].__name__}{line_style}>{edge[1].__name__}\n"
        text += "```"

        display(Markdown(text))
        return text

    def get_all_paths(self, in_type: Type[In], out_type: Type[Out], full: bool = False):
        if full:
            G = self.vG
        else:
            G = self.G

        if self.can_convert(in_type, out_type, full=full):
            for path in nx.shortest_simple_paths(G, in_type, out_type):
                converters = [
                    G.get_edge_data(path[i], path[i + 1])["converter"]
                    for i in range(len(path) - 1)
                ]
                if len(path) == 1 and len(converters) == 0:
                    yield path * 2, [lambda x: x]
                else:
                    yield path, converters


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

    def get_instance(self, instance: dict[Type[TypeVar], Type]):
        generic = self.origin
        args_list = []
        if self.args is None:
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


def infer_generic_type(type_var: Type[Any], instance: dict[Type[TypeVar], Type]):
    model = gen_typevar_model(type_var)
    return model.get_instance(instance)
