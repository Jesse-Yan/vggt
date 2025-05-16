#!/bin/bash

# 基础路径配置
DATASET_BASE_DIR="dataset/v2x_vit"
OUTPUT_BASE_DIR="bev_plots_v2xvit"
VISUALIZATION_SCRIPT="visualization.py" # 假设 visualization.py 在当前路径或已加入PATH
PYTHON_CMD="python"
VISUALIZATION_SCRIPT_CMD="$PYTHON_CMD $VISUALIZATION_SCRIPT"

# 创建总的输出基础目录
mkdir -p "$OUTPUT_BASE_DIR"
echo "总输出目录已创建/确认: $OUTPUT_BASE_DIR"

# 定义要处理的模式 (train, test, validate)
MODES=("train" "test" "validate")

# 遍历每种模式
for mode in "${MODES[@]}"; do
    CURRENT_DATASET_DIR="$DATASET_BASE_DIR/$mode"
    CURRENT_SAVE_DIR="$OUTPUT_BASE_DIR/$mode"

    echo ""
    echo "-----------------------------------------------------"
    echo "开始处理模式: $mode"
    echo "数据源目录: $CURRENT_DATASET_DIR"
    echo "将要保存到: $CURRENT_SAVE_DIR"
    echo "-----------------------------------------------------"

    # 检查模式的数据目录是否存在
    if [ ! -d "$CURRENT_DATASET_DIR" ]; then
        echo "警告: 目录 $CURRENT_DATASET_DIR 不存在，跳过此模式。"
        continue
    fi

    # 创建该模式对应的输出子目录
    mkdir -p "$CURRENT_SAVE_DIR"

    # 遍历该模式下的所有 scenario_id 文件夹
    # 使用 find 命令确保只处理目录
    find "$CURRENT_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r scenario_path; do
        # 从路径中提取 scenario_id (即文件夹名称)
        scenario_id=$(basename "$scenario_path")

        echo "  正在处理 Scenario ID: $scenario_id"

        # 构建并执行命令
        # --base_dir 应为包含当前 scenario_id 文件夹的父目录 (即 CURRENT_DATASET_DIR)
        # --save_dir_base 应为希望 visualization.py 在其中创建输出的目录 (即 CURRENT_SAVE_DIR)
        # 假设 visualization.py 会在 CURRENT_SAVE_DIR 下根据 scenario_id 创建子目录保存其结果
        
        echo "    执行命令: $VISUALIZATION_SCRIPT_CMD --scenario_id \"$scenario_id\" --base_dir \"$CURRENT_DATASET_DIR\" --save_dir_base \"$CURRENT_SAVE_DIR\""
        
        $VISUALIZATION_SCRIPT_CMD \
            --scenario_id "$scenario_id" \
            --base_dir "$CURRENT_DATASET_DIR" \
            --save_dir_base "$CURRENT_SAVE_DIR"

        if [ $? -eq 0 ]; then
            echo "    成功处理 Scenario ID: $scenario_id"
        else
            echo "    处理 Scenario ID: $scenario_id 时发生错误 (退出码: $?)"
        fi
        echo "  -----------------------------------"
    done

    echo "模式 $mode 处理完毕。"
done

echo ""
echo "所有模式处理完成。"