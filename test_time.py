import argparse
import json


def avg(values):
    return sum(values) / len(values) if values else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="解析 log 文件")
    parser.add_argument("log_path", type=str, help="log 文件路径")
    return parser.parse_args()

if __name__ == "__main__":
    decode_time = []
    rapido_time = []
    matchr_time = []

    args = parse_args()
    with open(args.log_path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    for line in data:
        if "Average Pre-Process Time" in line:
            try:
                value = float(line.split("Average Pre-Process Time:")[1].split(" ms")[0].strip())
                decode_time.append(value)
            except (IndexError, ValueError):
                continue
        elif "Average Inference Time" in line:
            try:
                value = float(line.split("Average Inference Time:")[1].split(" ms")[0].strip())
                rapido_time.append(value)
            except (IndexError, ValueError):
                continue
        elif "Average Post-Process Time" in line:
            try:
                value = float(line.split("Average Post-Process Time:")[1].split(" ms")[0].strip())
                matchr_time.append(value)
            except (IndexError, ValueError):
                continue

    result = {
        "preprocess_avg_ms": f"{avg(decode_time[1:]):.4f} ms",
        "preprocess_count": len(decode_time) - 1,
        "inference_avg_ms": f"{avg(rapido_time[1:]):.4f} ms",
        "inference_count": len(rapido_time) - 1,
        "postprocess_avg_ms": f"{avg(matchr_time[1:]):.4f} ms",
        "postprocess_count": len(matchr_time) - 1,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))