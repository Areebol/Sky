<!--
 * @Descripttion: 
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-04 20:49:30
-->
[TOC]

## Introduction
Django是一个开放源代码的Web应用框架，使用Python编写
采用MVT的软件设计模式

- 模型 Model
- 视图 View
- 模板 Template

只需要很少的代码就可以完成一个正式网站需要使用的大部分内容
本身基于MVC模型 -> model + view + controller

- 强大的数据库
- 后台功能

**操作流程**
![用户操作流程](https://www.runoob.com/wp-content/uploads/2020/05/1589777036-2760-fs1oSv4dOWAwC5yW.png)

- 浏览器向服务器发出请求request，request访问视图函数
- 不涉及数据调用，返回一个template
- 设计数据调用，调用模型，模型在数据库查找数据，将返回数据填充到模板，返回页面
  
## 创建项目
```python
django-admin startproject New_project
```

```bash
cd New_project
tree
|-- New_project
|   |-- __init__.py
|   |-- asgi.py
|   |-- settings.py
|   |-- urls.py
|   `-- wsgi.py
`-- manage.py
```

- New_project: 项目的容器。
- `manage.py`: 一个实用的命令行工具，可让你以各种方式与该 Django 项目进行交互。
- New_project/`__init__.py`: 一个空文件，告诉 Python 该目录是一个 Python 包。
- New_project/asgi.py: 一个 ASGI 兼容的 Web 服务器的入口，以便运行你的项目。
- New_project/settings.py: 该 Django 项目的设置/配置。
- New_project/urls.py: 该 Django 项目的 URL 声明; 一份由 Django 驱动的网站"目录"。
- New_project/wsgi.py: 一个 WSGI 兼容的 Web 服务器的入口，以便运行你的项目。

**运行服务器**
```bash
python manage.py runserver localhost:8000
```

**视图和url配置**

创建视图函数
`views.py`
```python
from django.http import HttpResponse
def hello(request):
    return HttpResponse("Hello world!")         
```

绑定url和视图函数
`urls.py`
```python
from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'^$', views.hello),
]
```
```python
# 使用其他方法绑定
from django.conf.urls import url
from . import views
urlpatterns = [
    path("hello/",views.hello),
]
```

`path(route, view, kwargs=None, name=None)`
- route: 字符串，表示 URL 规则，与之匹配的 URL 会执行对应的第二个参数 view。

- view: 用于执行与正则表达式匹配的 URL 请求。

- kwargs: 视图使用的字典类型的参数。

- name: 用来反向获取 URL。

## Django模板


## Django模型
Django对各种数据库提供了很好的支持，提供了统一的调用API

Object Relational Mapping 
ORM对应关系表
![p](https://www.runoob.com/wp-content/uploads/2020/05/orm-object.png)

`创建MySQL数据库`
```bash
create database 数据库名称 default charset=utf8; #防止编码问题，指定为 utf8
```

`在setting.py中配置数据库`
```python
DATABASES = { 
    'default': 
    { 
        'ENGINE': 'django.db.backends.mysql',    # 数据库引擎
        'NAME': 'db_name', # 数据库名称
        'HOST': '127.0.0.1', # 数据库地址，本机 ip 地址 127.0.0.1 
        'PORT': 3306, # 端口 
        'USER': 'root',  # 数据库用户名
        'PASSWORD': '', # 数据库密码
    }  
}
```

**定义模型**
创建APP
使用模型，首先必须要创建一个app
```bash
django-admin startapp TestModel
```

`文件目录`
```bash
HelloWorld
|-- HelloWorld
|-- manage.py
...
|-- TestModel
|   |-- __init__.py
|   |-- admin.py
|   |-- models.py
|   |-- tests.py
|   `-- views.py
```

`增加模型`
```python
from django.db import models
class Test(models.Model):
    name=models.CharField(max_length=20)
```
- 类名代表了数据库表名，继承了models.Model
- 字段为name，数据类型为CharField

`添加apps`
```python
INSTALLED_APPS = (
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'TestModel',               # 添加此项
)
```

`数据库迁移`
```bash
python mange.py makemigrations 
python mange.py migrate
```

**数据库操作**
`添加数据`
```python
from django.http import HttpResponse
from TestModel.models import Test
# 数据库操作
def testdb(request):
    test1 = Test(name='runoob')
    test1.save()
    return HttpResponse("<p>数据添加成功！</p>")
```

`获取数据`
```python
# 数据库操作
from django.http import HttpResponse
from TestModel.models import Test
def testdb(request):
    # 初始化
    response = ""
    response1 = ""
    # 通过objects这个模型管理器的all()获得所有数据行，相当于SQL中的SELECT * FROM
    list = Test.objects.all()
    # filter相当于SQL中的WHERE，可设置条件过滤结果
    response2 = Test.objects.filter(id=1) 
    # 获取单个对象
    response3 = Test.objects.get(id=1) 
    # 限制返回的数据 相当于 SQL 中的 OFFSET 0 LIMIT 2;
    Test.objects.order_by('name')[0:2]
    #数据排序
    Test.objects.order_by("id")
    # 上面的方法可以连锁使用
    Test.objects.filter(name="runoob").order_by("id")
    # 输出所有数据
    for var in list:
        response1 += var.name + " "
    response = response1
    return HttpResponse("<p>" + response + "</p>")
```

`更新数据`
```python
from django.http import HttpResponse
from TestModel.models import Test
# 数据库操作
def testdb(request):
    # 修改其中一个id=1的name字段，再save，相当于SQL中的UPDATE
    test1 = Test.objects.get(id=1)
    test1.name = 'Google'
    test1.save()
    # 另外一种方式
    Test.objects.filter(id=1).update(name='Google')
    # 修改所有的列
    Test.objects.all().update(name='Google') 
    return HttpResponse("<p>修改成功</p>")
```

`删除数据`
```python
from django.http import HttpResponse
from TestModel.models import Test
# 数据库操作
def testdb(request):
    # 删除id=1的数据
    test1 = Test.objects.get(id=1)
    test1.delete()
    
    # 另外一种方式
    Test.objects.filter(id=1).delete()
    # 删除所有数据
    Test.objects.all().delete()
    
    return HttpResponse("<p>删除成功</p>")
```

## 



