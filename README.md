<div align="center">

# TypeGraph

_**TypeGraph** is a Python library designed for type conversion between various types, including custom types, built-in types, and structural types (such as lists, sets, and dictionaries). It is also compatible with `Pydantic Annotated[T, Field(...)]`. The library supports both synchronous and asynchronous conversion methods._

> 蓦然回首，那人却在灯火阑珊处

 [![CodeFactor](https://www.codefactor.io/repository/github/luxuncang/typegraph/badge)](https://www.codefactor.io/repository/github/luxuncang/typegraph)
 [![GitHub](https://img.shields.io/github/license/luxuncang/typegraph)](https://github.com/luxuncang/typegraph/blob/master/LICENSE)
 [![CodeQL](https://github.com/luxuncang/typegraph/workflows/CodeQL/badge.svg)](https://github.com/luxuncang/typegraph/blob/master/.github/workflows/codeql.yml)

English | [简体中文](./README-zh.md)

</div>

## Features
- Register type converters for synchronous and asynchronous functions.
- Automatically convert function arguments based on type annotations.
- Support for subclass, union types, and structural types conversion.
- Recursive Generic Calculation.
- Visualize the conversion graph using mermaid syntax.

## Installation
Install the required dependencies with the following command:

```sh
pip install typegraph3
```

Or

```sh
pdm add typegraph3
```

## Getting Started

### Example: Synchronous Converter
Register and use a synchronous converter:

```python
from typegraph import PdtConverter

converter = PdtConverter()

@converter.register_converter(int, str)
def int_to_str(value: int) -> str:
    return str(value)

result = converter.convert(10, str)  # "10"
print(result)
```

### Example: Asynchronous Converter
Register and use an asynchronous converter:

```python
import asyncio
from typegraph import PdtConverter

converter = PdtConverter()

@converter.async_register_converter(str, int)
async def str_to_int(value: str) -> int:
    return int(value)

async def test_async_conversion():
    result = await converter.async_convert("10", int)  # 10
    print(result)

asyncio.run(test_async_conversion())
```

### Example: Protocol Types

```python
from typing import Protocol, TypedDict, runtime_checkable
from dataclasses import dataclass

from typegraph import PdtConverter

t = PdtConverter()

class Person(Protocol):
    name: str
    phone: str
    address: str

    def get_name(self) -> str:
        ...

class PersonDict(TypedDict):
    name: str
    phone: str
    address: str

class A:
    name: str
    phone: str
    address: str

    def __init__(self, name: str, phone: str, address: str):
        self.name = name
        self.phone = phone
        self.address = address

    def get_name(self) -> str:
        return self.name

@dataclass
class B:
    name: str
    phone: str
    address: str

@t.register_converter(dict, PersonDict)
def convert_dict_to_persondict(data: dict):
    return PersonDict(
        name=data["name"],
        phone=data["phone"],
        address=data["address"]
    )

@t.register_converter(Person, str)
def convert_person_to_str(data: Person):
    return f"{data.name} {data.phone} {data.address}"

@t.register_converter(dict, A)
def convert_dict_to_a(data: dict):
    return A(data["name"], data["phone"], data["address"])

@t.register_converter(dict, B)
def convert_dict_to_b(data: dict):
    return B(data["name"], data["phone"], data["address"])

@t.auto_convert()
def test(a: str):
    return a

d = {"name": "John", "phone": "123", "address": "123"}

t.convert([d], list[str], debug=True)
```

`t.show_mermaid_graph()`

```mermaid
graph TD;
dict-->PersonDict
dict-->A
dict-->B
Person-->str
```

`t.show_mermaid_graph()`

```mermaid
graph TD;
dict-->PersonDict
dict-->A
dict-->B
Person-->str
A-.->Person
```

```bash
Converting dict[str, str] to <class 'str'> using [<class 'dict'>, <class '__main__.A'>, <class '__main__.Person'>, <class 'str'>], <function TypeConverter.get_converter.<locals>.<lambda>.<locals>.<lambda> at 0x7f1f3306fac0>

['John 123 123']
```

### Recursive Generic Calculation
 
> Default recursion depth is two

```python
from typing import Iterable, TypeVar, Annotated

from pydantic import Field

from typegraph import PdtConverter

t = PdtConverter()

K = TypeVar("K")
V = TypeVar("V")

P = Annotated[int, Field(ge=0, le=10)]

@t.register_generic_converter(dict[K, V], dict[V, K])
def convert_dict_to_dict(value: dict[K, V]) -> dict[V, K]:
    return {v: k for k, v in value.items()}

@t.register_generic_converter(V, Iterable[V])
def convert_to_iterable(value: V) -> Iterable[V]:
    return [value]

@t.register_converter(P, int)
def convert_p_to_int(value: P) -> int:
    return value

try:
    t.convert(11, P)
except Exception as e:
    print(e)
```

**Pydantic Annotated[T, Feild(...)]**

```bash
No converter registered for <class 'int'> to typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]
```

```python
t.convert(5, P)
```

```bash
5
```
**dict[K,V]->dict[V,K]**

```python
t.convert({1: "2", 3: "4"}, dict[int, int], debug=True)
```

```bash
Converting dict[int, str] to dict[str, int] using [dict[int, str], dict[str, int]], <function convert_dict_to_dict at 0x7f18e595c3a0>
{'2': 1, '4': 3}
```

**V->Iterable[V]**

```python
t.convert(1, Iterable[Iterable[Iterable[P]]], debug=True)
```

```bash
Converting typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])] to typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]] using [typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])], typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]], typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]], typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]], <function PdtConverter.get_converter.<locals>.<lambda>.<locals>.<lambda> at 0x7f18e46ecf70>
[[[1]]]
```

**Visualization**

```python
t.show_mermaid_graph()
```

```mermaid
graph TD;
node0["typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]"] --> node1["int"]
node0["typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]"] -.-> node2["typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]"]
node1["int"] -.-> node3["typing.Iterable[int]"]
node3["typing.Iterable[int]"] -.-> node4["typing.Iterable[typing.Iterable[int]]"]
node2["typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]"] -.-> node5["typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]"]
node5["typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]"] -.-> node6["typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]"]
node7["dict[int, str]"] -.-> node8["dict[str, int]"]
node7["dict[int, str]"] -.-> node9["typing.Iterable[dict[int, str]]"]
node8["dict[str, int]"] -.-> node7["dict[int, str]"]
node8["dict[str, int]"] -.-> node10["typing.Iterable[dict[str, int]]"]
node9["typing.Iterable[dict[int, str]]"] -.-> node11["typing.Iterable[typing.Iterable[dict[int, str]]]"]
node10["typing.Iterable[dict[str, int]]"] -.-> node12["typing.Iterable[typing.Iterable[dict[str, int]]]"]
node6["typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]"] -.-> node13["typing.Iterable[typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]]"]
node13["typing.Iterable[typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]]"] -.-> node14["typing.Iterable[typing.Iterable[typing.Iterable[typing.Iterable[typing.Iterable[typing.Annotated[int, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=10)])]]]]]]"]
```

### Auto-Convert Decorator
Automatically convert function arguments based on type annotations:

#### Synchronous

```python
from typegraph import PdtConverter

converter = PdtConverter()

@converter.register_converter(str, int)
def str_to_int(value: str) -> int:
    return int(value)

@converter.auto_convert()
def add_one(x: int) -> int:
    return x + 1

result = add_one("10")  # 11
print(result)
```

#### Asynchronous

```python
from typegraph import PdtConverter
import asyncio

converter = PdtConverter()

@converter.async_register_converter(str, int)
async def str_to_int(value: str) -> int:
    return int(value)

@converter.async_auto_convert()
async def add_one(x: int) -> int:
    return x + 1

async def test_async():
    result = await add_one("10")  # 11
    print(result)

asyncio.run(test_async())
```

## Testing

Unit tests are provided to ensure the library functions correctly. Run the tests:

```bash
pdm test
```

Tests cover:
- Registration and execution of synchronous converters.
- Registration and execution of asynchronous converters.
- Conversion capability checks.
- Automatic conversion of function arguments (both synchronous and asynchronous).

## Visualization

You can visualize the type conversion graph:

```python
from typegraph import PdtConverter

t = PdtConverter()

class Test:
    def __init__(self, t):
        self.t = t

@t.register_converter(float, Test)
def str_to_Test(input_value):
    return Test(input_value)

@t.register_converter(Test, float)
def B_to_float(input_value):
    return float(input_value.t)

@t.register_converter(float, str)
async def float_to_str(input_value):
    return str(input_value)

t.show_mermaid_graph()
```

```mermaid
graph TD;
float-->Test
float-->str
Test-->float
```

The graph will be displayed using mermaid syntax, which can be rendered online or in supported environments like Jupyter Notebooks.

## Supported Types

- [X] Subclass type
- [X] Union type
- [X] Annotated type `Pydantic Annotated[T, Feild(...)]`
- [X] Structural type
- [X] Protocol type
- [X] TypedDict type
- [X] Generic type
- [X] Dataclass (Dataclass/BaseModel)

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact
If you have any questions or concerns, please open an issue in this repository.