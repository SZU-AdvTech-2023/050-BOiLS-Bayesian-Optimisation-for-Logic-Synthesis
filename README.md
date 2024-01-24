# 基于iMap的智能组合逻辑优化与工艺映射

系统环境：Linux

软件：python3.7、anaconda

```bash
#下载iMap
git clone https://gitee.com/oscc-project/iMAP.git

#下载NSGA2与Boils
git clone https://gitee.com/wang-rui13132008217/nsga2.git
#将NSGA2下的文件移动到iMAP/ai_infra目录下
mv nsga2/* iMap/ai_infra

cd iMap
#编译imap
mkdir build && cd build
cmake ..
make -j 8

#复制imap执行文件至ai_infra目录下
cd ..
cp bin/imap ai_infra
```

修改文件aig文件与结果存储路径：

修改results_storage_root_path.txt内为ai_infra文件夹所在路径。

运行utils/utils_save.py文件，保存路径。

创建python3.7虚拟环境

``` bash
conda create -n yourEnv python=3.7 #创建python3.7虚拟环境
conda activate yourEnv	#激活环境
pip install -r requirements.txt #下载必要库
```
## NSGA2运行示例：

路径进入GA目录，运行示例：

```bash
python main_nsga2.py --designs_group_id arbiter --n_gen 100 --pop_size 20 --seq_length 10 --seed 0
```

## Boils运行示例：

路径进入Boils目录，运行示例：

```bash
python main_boils.py --designs_group_id arbiter --seq_length 10 --device -1 --seed 0
```

