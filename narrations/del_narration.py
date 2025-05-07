import json

with open('narration_with_gaze_600s.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

new_data = {}
for key, value in data.items():
    # 假设每个 value 都有 narration_pass_1 和 narrations
    narrations = value.get('narration_pass_1', {}).get('narrations', [])
    new_data[key] = {'narrations': narrations}

with open('ego4d_narration.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)