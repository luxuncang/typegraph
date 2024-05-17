
# from typing import TypeVar, Generic, Union
# from typing_extensions import get_args, get_origin, get_original_bases, get_protocol_members, reveal_type
# from typing_inspect import is_union_type, get_generic_type
# # from pytypes import deep_type

# a = [1, 2, 3]

# T = TypeVar('T', covariant=True)

# class Container(Generic[T]):
#     def __init__(self, value: T):
#         self._value = value

#     def get(self) -> T:
#         return self._value

# # reveal_type(a)  # Revealed type is 'Tuple[int, int, int]'

# def get_type(obj):
#     print("type(obj):", type(obj))
#     print("get_origin(type(obj)):", get_origin(type(obj)))
#     print("reveal_type(obj):", reveal_type(obj))

# print(list[int] == list)
# get_type(Container(1))
# # print(deep_type(a))
# tpl = tuple([str, int])
# print(list[Union[tpl]] == list[Union[int, str]])
from __future__ import annotations
from pprint import pprint
import typing
import types
from typing import TypeVar, Generic, Union, Annotated, List, Dict, Callable, Any, TypedDict
from typing_extensions import get_original_bases, get_args, Unpack
from typing_inspect import is_callable_type, get_generic_type, get_generic_bases, get_parameters
# from pytypes import deep_type

from typegraph.core import gen_typevar_model, infer_generic_type, generate_type, get_all_subclasses
from typegraph.type_utils import get_origin, deep_type, is_structural_type

T = TypeVar('T')
K = TypeVar('K')

a = List[T]
b = list[Union[int, a]]
c =dict[K, b]
e = Callable[[List[T]], T]

class A(Generic[T]):
    ...

# pprint(infer_generic_type(e , {T: A}) == Callable[[List[A]], A])
# pprint(gen_typevar_model(e))

# pprint(List[int].__dict__)
# pprint(list[int].__dict__)
# print(get_args(List[list]))
# print(get_original_bases(List[int]))
# a = types.GenericAlias(list, int)
# print(a == list[int])
# print(get_origin(List[int]))

T = TypeVar('T')


class B(A):
    ...

# # print(get_origin(A))
# print(deep_type([1,2,3]))

# print(Annotated[int, {'a': 1, 'b': 2}].__dict__)


l = {"a": 1, "b": 2, "c": 3, "d": A[int]()}

class Movie(TypedDict):
    name: str
    year: int

def lfun(a: int, b: str, c) -> A[int]:
    ...

cc = lfun

def w(func: Callable[[Unpack[Movie]], A[int]]):
    ...

# w(lfun)

# print(deep_type(lfun))
# get_generic_bases(B())

# print(type(A[int]()))

# case1 A -> B, List[A] -> List[B]
# case2 A -> B, C -> B, List[A | C] -> List[B]
# case3 A -> B, List[A] -> List[C | B]
# case4 A -> B, 

# print(is_structural_type(list[int]))

def sync_iter(input_type, out_type):
    from typing import get_origin
    if is_structural_type(input_type) and is_structural_type(out_type):
        if get_origin(input_type) == get_origin(out_type):
            yield get_args(input_type), get_args(out_type)


for a,b in sync_iter(List[int], list[str]):
    print(a, b)