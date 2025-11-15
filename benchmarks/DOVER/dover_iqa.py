import os
import subprocess
import re
import json

def evaluate_video(video_path, opt_path="./dover.yml", device="cuda", fusion=False):
    """
    评估视频的质量分数
    
    参数:
        video_path (str): 视频文件的路径
        opt_path (str): DOVER配置文件的路径
        device (str): 运行设备 ('cuda' 或 'cpu')
        fusion (bool): 是否输出融合分数
        
    返回:
        dict: 包含评估结果的字典
    """
    # 确保视频文件存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 确保配置文件存在
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"配置文件不存在: {opt_path}")
    
    # 构建命令行参数
    cmd = [
        "python", 
        "evaluate_one_video.py",
        "-v", video_path,
        "-o", opt_path,
        "-d", str(device)
    ]
    
    if fusion:
        cmd.append("-f")
    
    # 执行命令并捕获输出
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"评估失败，错误码: {e.returncode}, 错误信息: {e.stderr}")
    
    # 解析输出
    return _parse_output(output, video_path, fusion)

def _parse_output(output, video_path, fusion):
    """解析评估脚本的输出并返回结构化结果"""
    result = {
        "video_path": video_path,
        "raw_output": output
    }
    
    if fusion:
        # 解析融合分数
        fusion_match = re.search(r"Normalized fused overall score \(scale in \[0,1\]\): ([-+]?\d*\.\d+|\d+)", output)
        if fusion_match:
            result["fused_score"] = float(fusion_match.group(1))
    else:
        # 解析技术质量和美学质量分数
        result["dataset_comparisons"] = {}
        
        # 查找所有数据集比较
        dataset_patterns = [
            (r"Compared with all videos in the ([\w_-]+) dataset:", "dataset"),
            (r"the technical quality of video .+ is better than (\d+)% of videos, with normalized score ([-+]?\d*\.\d+|\d+)", "technical"),
            (r"the aesthetic quality of video .+ is better than (\d+)% of videos, with normalized score ([-+]?\d*\.\d+|\d+)", "aesthetic")
        ]
        
        current_dataset = None
        
        for line in output.split('\n'):
            for pattern, key in dataset_patterns:
                match = re.search(pattern, line)
                if match:
                    if key == "dataset":
                        current_dataset = match.group(1)
                        result["dataset_comparisons"][current_dataset] = {}
                    elif key == "technical" and current_dataset:
                        result["dataset_comparisons"][current_dataset]["technical_quality"] = {
                            "percentile": int(match.group(1)),
                            "normalized_score": float(match.group(2))
                        }
                    elif key == "aesthetic" and current_dataset:
                        result["dataset_comparisons"][current_dataset]["aesthetic_quality"] = {
                            "percentile": int(match.group(1)),
                            "normalized_score": float(match.group(2))
                        }
    
    return result

def evaluate_video_json(video_path, opt_path="./dover.yml", device="cuda", fusion=False):
    """
    评估视频质量并返回JSON格式的结果
    
    参数:
        video_path (str): 视频文件的路径
        opt_path (str): DOVER配置文件的路径
        device (str): 运行设备 ('cuda' 或 'cpu')
        fusion (bool): 是否输出融合分数
        
    返回:
        str: JSON格式的评估结果
    """
    result = evaluate_video(video_path, opt_path, device, fusion)
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 简单的命令行接口，用于测试
    import argparse
    
    parser = argparse.ArgumentParser(description="DOVER视频质量评估工具")
    parser.add_argument("video_path", type=str, help="视频文件路径")
    parser.add_argument("-o", "--opt", type=str, default="./dover.yml", help="配置文件路径")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("-f", "--fusion", action="store_true", help="是否输出融合分数")
    parser.add_argument("-j", "--json", action="store_true", help="以JSON格式输出结果")
    
    args = parser.parse_args()
    
    if args.json:
        print(evaluate_video_json(args.video_path, args.opt, args.device, args.fusion))
    else:
        result = evaluate_video(args.video_path, args.opt, args.device, args.fusion)