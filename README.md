# TypeGraph3

## 概述
**类型转换器 (TypeConverter)** 是一个 Python 库，用于在不同类型之间进行类型转换，包括自定义类型、内置类型和结构类型（如列表、集合和字典）。它支持同步和异步的转换方法。

## 功能
- 注册同步和异步函数的类型转换器。
- 根据类型注解自动转换函数参数。
- 支持子类、联合类型和结构类型的转换。
- 使用 mermaid 语法可视化转换图。

## 安装
使用以下命令安装运行该库所需的依赖项：

```sh
pip install typegraph3
```

## 入门指南

### 示例：同步转换器
注册同步转换器并使用：

```python
from typegraph3 import TypeConverter

converter = TypeConverter()

@converter.register_converter(int, str)
def int_to_str(value: int) -> str:
    return str(value)

result = converter.convert(10, str)  # "10"
print(result)
```

### 示例：异步转换器
注册异步转换器并使用：

```python
import asyncio
from typegraph3 import TypeConverter

converter = TypeConverter()

@converter.async_register_converter(str, int)
async def str_to_int(value: str) -> int:
    return int(value)

async def test_async_conversion():
    result = await converter.async_convert("10", int)  # 10
    print(result)

asyncio.run(test_async_conversion())
```

### 自动转换装饰器
根据类型注解自动转换函数参数：

#### 同步

```python
from typegraph3 import TypeConverter

converter = TypeConverter()

@converter.register_converter(str, int)
def str_to_int(value: str) -> int:
    return int(value)

@converter.auto_convert()
def add_one(x: int) -> int:
    return x + 1

result = add_one("10")  # 11
print(result)
```

#### 异步

```python
from typegraph3 import TypeConverter
import asyncio

converter = TypeConverter()

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

## 测试

提供了单元测试，以确保库的正确功能。运行测试：

```bash
pytest test_switch.py
```

测试覆盖了：
- 同步转换器的注册与执行。
- 异步转换器的注册与执行。
- 转换能力检查。
- 函数参数的自动转换（同步和异步）。

## 可视化

您可以可视化类型转换图：

```python
from typegraph3 import TypeConverter

t = TypeConverter()

class Test:
    def __init__(self, t):
        self.t = t

class TestFloat(float):
    ...


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
TestFloat-.->float
```

图将使用 mermaid 语法显示，您可以在线渲染或在支持的环境中（如 Jupyter Notebooks）进行渲染。

## 支持的类型
- [X] 子类类型 (SubClass type)
- [X] 联合类型 (Union type)
- [X] 注解类型 (Annotated type)
- [X] 结构类型 (Structural type)
- [ ] 泛型类型 (Generic type)

## 许可
此项目使用 MIT 许可证。

## 贡献
欢迎贡献！请提出 issue 或提交 pull request 来进行更改。

## 联系方式
如果您有任何问题或疑问，请在此仓库中提出 issue。
