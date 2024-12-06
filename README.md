# 简介
用PBD进行软体仿真

采用距离约束和体积约束两种约束

####  距离约束
![](photo/Distance%20Constraint.png)

#### 体积约束
![](photo/Volume%20Conservation%20Constraint%201.png)
![](photo/Volume%20Conservation%20Constraint%202.png)

# 参考
https://github.com/chunleili/tiPBD

https://matthias-research.github.io/pages/tenMinutePhysics/10-softBodies.pdf

https://matthias-research.github.io/pages/tenMinutePhysics/09-xpbd.pdf

https://matthias-research.github.io/pages/tenMinutePhysics/

# 运行方式
安装taichi，当前使用版本是1.7.2。运行
```
python PBD.py
``` 

# 运行结果
![](results/video.gif)

# 代码说明
可修改<code>ReadMesh.py</code>中的<code>data</code>来实现不同物体的仿真

<code>PBD.py</code>中的<code>is_record</code>设置为<code>True</code>，将会把运行结果保存到<code>results</code>文件夹下