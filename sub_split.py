#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def split_video(input_path: Path, output_dir: Path, max_duration: int, dry_run: bool):
    """分割单个视频文件"""
    try:
        with VideoFileClip(str(input_path)) as video:
            total_duration = video.duration
            clip_num = int(total_duration // max_duration) + (1 if total_duration % max_duration > 0 else 0)
            
            if dry_run:
                print(f"[DRY RUN] {input_path.name} -> {clip_num} clips")
                return clip_num

            for i in range(clip_num):
                start = i * max_duration
                end = min((i + 1) * max_duration, total_duration)
                output_path = output_dir / f"{input_path.stem}_part{i+1:03d}{input_path.suffix}"
                
                video.subclip(start, end).write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio_codec="aac",
                    threads=4,
                    logger=None,
                    preset="fast"  # 编码速度/质量平衡
                )
            return clip_num
            
    except Exception as e:
        print(f"Error processing {input_path.name}: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description="Split MP4 videos into segments by maximum duration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing MP4 files")
    parser.add_argument("-o", "--output-dir", default="split_output", help="Output directory")
    parser.add_argument("-m", "--max-duration", type=int, default=19, 
                        help="Maximum duration per segment (seconds)")
    parser.add_argument("-d", "--dry-run", action="store_true", 
                        help="Simulate without actual processing")
    parser.add_argument("--keep-original", action="store_true", 
                        help="Keep original files after splitting")
    args = parser.parse_args()

    # 初始化路径
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理文件
    video_files = list(input_dir.glob("*.mp4"))
    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return

    print(f"Processing {len(video_files)} files in {input_dir}")
    total_clips = 0

    for video in tqdm(video_files):
        clips = split_video(video, output_dir, args.max_duration, args.dry_run)
        total_clips += clips
        if not args.dry_run and clips > 0 and not args.keep_original:
            video.unlink()  # 删除原文件

    print(f"\nCompleted: {total_clips} clips generated in {output_dir}")

if __name__ == "__main__":
    main()
