import json

files_to_merge = ["causal_egtea.json", "spatial_egtea.json", "temporal_egtea.json"]
merged_output_file = "egtea.json"

merged_data = []

for file_name in files_to_merge:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        merged_data.extend(data)

# 保存合并后的数据
with open(merged_output_file, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"✅ 合并完成！保存到了 {merged_output_file}")