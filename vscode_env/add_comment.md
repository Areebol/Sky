<!--
 * @Descripttion: your project
 * @version: 1.0
 * @Author: Areebol
 * @Date: 2023-06-03 21:18:06
 * @LastEditors: Aidam_Bo
 * @LastEditTime: 2023-06-03 21:41:09
-->

# vscode在文件顶部添加作者，时间和注释信息

- 安装创建KoroFileHeader
- 在管理-设置-"fileheader" 在setting.json中编辑
- 放入头部注释代码

```
// 文件头部注释
 
"fileheader.customMade": {
    "Descripttion":"your project",
    "version":"1.0",
    "Author":"Areebol",
    "Date":"Do not edit",
    "LastEditTime":"Do not Edit"
},
 
 
//函数注释
"fileheader.cursorMode": {
    "name":"",
    "msg":"",
    "param":"",
    "return":""
},
```

- 快捷键：ctrl + alt + i 生成头部注释

- 快捷键：ctrl + alt + t 生成函数注释

ctrl+alt+i,添加 文件头 注释
ctrl+alt+t,添加 函数 注释(默认,但是可能和toggle integrated terminal)