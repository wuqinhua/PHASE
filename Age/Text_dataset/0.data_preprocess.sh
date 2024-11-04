#!/bin/bash

src_dir="/data/wuqinhua/phase/age/test"
dst_dir="/data/wuqinhua/phase/age/test/raw"

mkdir -p "/data/wuqinhua/phase/age/test/raw"

find "$src_dir" -type f | while read filepath; do
  filename=$(basename "$filepath")
  middle_12_14=$(expr substr "$filename" 12 3)
  new_filename=$(expr substr "$filename" 16 $((${#filename} - 15)))
  echo "Char(12-14): $middle_12_14, new_name: $new_filename"
  target_dir="$dst_dir/$middle_12_14"
  mkdir -p "$target_dir"
  mv "$filepath" "$target_dir/"
  moved_file="$target_dir/$filename"
  mv "$moved_file" "$target_dir/$new_filename"
done

echo "File move and rename complete."