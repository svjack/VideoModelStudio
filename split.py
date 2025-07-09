import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from scenedetect import detect, ContentDetector, SceneManager, open_video
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm

def process_video(video_path: Path, output_dir: Path, enable_splitting: bool) -> int:
    """
    处理单个视频文件，检测场景并分割
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        enable_splitting: 是否启用场景分割
    Returns:
        检测到的场景数量
    """
    try:
        # 检测场景
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video, show_progress=False)
        scenes = scene_manager.get_scene_list()
        num_scenes = len(scenes)

        if not scenes or not enable_splitting:
            # 不分割或没有检测到场景
            output_path = output_dir / video_path.name
            shutil.copy2(video_path, output_path)
            return 1 if not scenes else 0
        else:
            # 分割视频
            base_name = video_path.stem
            output_template = str(output_dir / f"{base_name}_scene_$SCENE_NUMBER.mp4")
            split_video_ffmpeg(
                str(video_path),
                scenes,
                output_file_template=output_template,
                show_progress=True
            )
            return num_scenes
            
    except Exception as e:
        print(f"处理视频 {video_path.name} 时出错: {str(e)}", file=sys.stderr)
        return 0

def process_zip(zip_path: Path, output_dir: Path, enable_splitting: bool) -> None:
    """
    处理ZIP文件中的所有MP4视频
    Args:
        zip_path: ZIP文件路径
        output_dir: 输出目录
        enable_splitting: 是否启用场景分割
    """
    # 创建临时解压目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 解压ZIP文件[1,5](@ref)
        print(f"解压文件: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_path)
        
        # 处理所有MP4文件
        video_count = 0
        scene_count = 0
        for video_file in tqdm(temp_path.glob("**/*.mp4")):
            if video_file.is_file():
                video_count += 1
                scenes = process_video(video_file, output_dir, enable_splitting)
                scene_count += scenes
                print(f"处理完成: {video_file.name} -> {scenes} 个场景")
        
        print(f"\n处理摘要:")
        print(f" - 解压视频文件: {video_count} 个")
        print(f" - 生成视频片段: {scene_count} 个")
        print(f" - 输出目录: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分割ZIP文件中的MP4视频")
    parser.add_argument("zip_file", help="输入的ZIP文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--no-split", action="store_false", dest="enable_splitting",
                        help="禁用场景分割（直接复制视频）")
    
    args = parser.parse_args()
    
    zip_path = Path(args.zip_file)
    output_dir = Path(args.output_dir)
    
    if not zip_path.exists():
        print(f"错误: ZIP文件不存在 {zip_path}", file=sys.stderr)
        sys.exit(1)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    process_zip(zip_path, output_dir, args.enable_splitting)
