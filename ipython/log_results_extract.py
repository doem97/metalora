import csv
import os
import re
from io import StringIO


def extract_info(log_content):
    """Extract accuracy information from log content."""
    many_med_few = re.search(
        r"\* many: (\d+\.\d+)%\s+med: (\d+\.\d+)%\s+few: (\d+\.\d+)%",
        log_content,
    )
    average = re.search(r"\* average: (\d+\.\d+)%", log_content)

    if many_med_few and average:
        return many_med_few.groups() + (average.group(1),)
    return None


def process_logs(root_dir, identifiers=None):
    """Process all successful logs in the directory that match any of the given identifiers.
    Args:
        root_dir: Root directory to search
        identifiers: Single string or list of strings to filter paths, or None to include all
    """
    results = []
    if isinstance(identifiers, str):
        identifiers = [identifiers]

    for root, _, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)

        # Skip if identifiers are specified and none match the current path
        if identifiers and not all(ident in rel_path for ident in identifiers):
            continue

        log_files = [f for f in files if f.endswith(".log")]
        if not log_files:
            continue

        # Sort log files by timestamp in filename (newest first)
        log_files.sort(
            key=lambda x: (
                re.findall(r"log-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", x)[0]
                if re.findall(r"log-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", x)
                else ""
            ),
            reverse=True,
        )

        for log_file in log_files:
            file_path = os.path.join(root, log_file)
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    if "Main process completed successfully" in content:
                        info = extract_info(content)
                        if info:
                            results.append((rel_path, log_file) + info)
                            break
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    return results


dataset = "dota"
model = "in21k_vit_b16_1024"
root_dir = f"/workspace/dso/transfm/metalora/output/{dataset}/{model}"
identifiers = ["ConstantLR", "adapter_dim_512"]
results = process_logs(root_dir, identifiers)

csv_output = StringIO()
csv_writer = csv.writer(csv_output)
csv_writer.writerow(["Expriment", "Log File", "Many", "Med", "Few", "Average"])
csv_writer.writerows(sorted(results, key=lambda x: x[0]))  # 按实验名称排序

print(csv_output.getvalue())
