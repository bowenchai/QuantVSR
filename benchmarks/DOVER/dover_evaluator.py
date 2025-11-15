# File: dover_evaluator.py

import os
original_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))

def evaluate_and_print_dover(video_path):
    os.chdir(script_dir)
    from DOVER import dover_iqa 
    result = dover_iqa.evaluate_video(video_path)
    fusion_result = dover_iqa.evaluate_video(video_path, fusion=True)
    print(f"fused_score: {fusion_result['fused_score']}")
    os.chdir(original_dir)
    return fusion_result['fused_score']

# If you want the script to be runnable directly as well
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        evaluate_and_print_dover(video_path)
    else:
        print("Usage: python dover_evaluator.py <video_path>")