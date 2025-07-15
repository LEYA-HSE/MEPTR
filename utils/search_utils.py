# coding: utf-8

import copy, os
import logging
import numpy as np
import datetime
from itertools import product
from typing import Any

def format_result_box_dual(step_num, param_name, candidate, fixed_params, dev_metrics, test_metrics, is_best=False):
    title = f"Шаг {step_num}: {param_name} = {candidate}"
    fixed_lines = [f"{k} = {v}" for k, v in fixed_params.items()]

    def format_metrics_block(metrics, label):
        lines = [f"  Результаты ({label.upper()}):"]
        for k in ["uar", "mF1", "ACC", "CCC"]:
            if k in metrics:
                val = metrics[k]
                line = f"    {k.upper():12} = {val:.4f}" if isinstance(val, float) else f"    {k.upper():12} = {val}"
                if is_best and label.lower() == "dev" and k.lower() == "mf1":
                    line += " ✅"
                lines.append(line)
        return lines

    content_lines = [title, "  Фиксировано:"]
    content_lines += [f"    {line}" for line in fixed_lines]

    content_lines += format_metrics_block(dev_metrics, "dev")
    content_lines.append("")
    content_lines += format_metrics_block(test_metrics, "test")

    max_width = max(len(line) for line in content_lines)
    border_top = "┌" + "─" * (max_width + 2) + "┐"
    border_bot = "└" + "─" * (max_width + 2) + "┘"

    box = [border_top]
    for line in content_lines:
        box.append(f"│ {line.ljust(max_width)} │")
    box.append(border_bot)

    return "\n".join(box)


def greedy_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    default_values: dict[str, Any],
    csv_prefix: str = None
):
    current_best_params = copy.deepcopy(default_values)
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Жадный (поэтапный) перебор гиперпараметров (Dev-based) ===\n")
        f.write(f"Модель: {model_name}\n")

    for i, param_name in enumerate(all_param_names):
        candidates = param_grid[param_name]
        tried_value = current_best_params[param_name]

        if i == 0:
            candidates_to_try = candidates
        else:
            candidates_to_try = [v for v in candidates if v != tried_value]

        best_val_for_param = tried_value
        best_metric_for_param = float("-inf")

        # Evaluate default value (only from 2nd step onward)
        if i != 0:
            config_default = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config_default, k, v)
            logging.info(f"[ШАГ {i+1}] {param_name} = {tried_value} (ранее проверенный)")

            combo_dir = os.path.join(base_config.checkpoint_dir, f"greedy_{param_name}_{tried_value}")
            os.makedirs(combo_dir, exist_ok=True)
            config_default.checkpoint_dir = combo_dir

            dev_metrics_default, test_metrics_default = train_fn(
                config_default,
                train_loader,
                dev_loader,
                test_loader,
            )

            dev_score = dev_metrics_default.get("mF1", 0)

            box_text = format_result_box_dual(
                step_num=i+1,
                param_name=param_name,
                candidate=tried_value,
                fixed_params={k: v for k, v in current_best_params.items() if k != param_name},
                dev_metrics=dev_metrics_default,
                test_metrics=test_metrics_default,
                is_best=True
            )

            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")

            _log_dataset_metrics(dev_metrics_default, overrides_file, label="dev")
            _log_dataset_metrics(test_metrics_default, overrides_file, label="test")

            best_metric_for_param = dev_score

        for candidate in candidates_to_try:
            config = copy.deepcopy(base_config)
            for k, v in current_best_params.items():
                setattr(config, k, v)
            setattr(config, param_name, candidate)
            logging.info(f"[ШАГ {i+1}] {param_name} = {candidate}, (остальные {current_best_params})")


            dev_metrics, test_metrics = train_fn(
                config,
                train_loader,
                dev_loader,
                test_loader,
            )

            dev_score = dev_metrics.get("mF1", 0)
            is_better = dev_score > best_metric_for_param

            box_text = format_result_box_dual(
                step_num=i+1,
                param_name=param_name,
                candidate=candidate,
                fixed_params={k: v for k, v in current_best_params.items() if k != param_name},
                dev_metrics=dev_metrics,
                test_metrics=test_metrics,
                is_best=is_better
            )

            with open(overrides_file, "a", encoding="utf-8") as f:
                f.write("\n" + box_text + "\n")

            _log_dataset_metrics(dev_metrics, overrides_file, label="dev")
            _log_dataset_metrics(test_metrics, overrides_file, label="test")

            if is_better:
                best_val_for_param = candidate
                best_metric_for_param = dev_score

        current_best_params[param_name] = best_val_for_param
        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write(f"\n>> [Итог Шаг{i+1}]: Лучший {param_name}={best_val_for_param}, dev_mF1={best_metric_for_param:.4f}\n")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Итоговая комбинация (Dev-based) ===\n")
        for k, v in current_best_params.items():
            f.write(f"{k} = {v}\n")

    logging.info("Готово! Лучшие параметры подобраны.")


def exhaustive_search(
    base_config,
    train_loader,
    dev_loader,
    test_loader,
    train_fn,
    overrides_file: str,
    param_grid: dict[str, list],
    csv_prefix: str = None
):
    all_param_names = list(param_grid.keys())
    model_name = getattr(base_config, "model_name", "UNKNOWN_MODEL")

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("=== Полный перебор гиперпараметров (Dev-based) ===\n")
        f.write(f"Модель: {model_name}\n")

    best_config = None
    best_score = float("-inf")
    best_metrics = {}
    combo_id = 0

    for combo in product(*(param_grid[param] for param in all_param_names)):
        combo_id += 1
        param_combo = dict(zip(all_param_names, combo))

        config = copy.deepcopy(base_config)
        for k, v in param_combo.items():
            setattr(config, k, v)

        logging.info(f"\n[Комбинация #{combo_id}] {param_combo}")

        # создаём уникальную папку для чекпоинтов текущей комбинации
        combo_dir = os.path.join(base_config.checkpoint_dir, f"combo_{combo_id}")
        os.makedirs(combo_dir, exist_ok=True)
        config.checkpoint_dir = combo_dir

        dev_metrics, test_metrics = train_fn(
            config,
            train_loader,
            dev_loader,
            test_loader,
        )

        dev_score = dev_metrics.get("mF1", 0)
        is_better = dev_score > best_score

        box_text = format_result_box_dual(
            step_num=combo_id,
            param_name=" + ".join(all_param_names),
            candidate=str(combo),
            fixed_params={},
            dev_metrics=dev_metrics,
            test_metrics=test_metrics,
            is_best=is_better
        )

        with open(overrides_file, "a", encoding="utf-8") as f:
            f.write("\n" + box_text + "\n")

        _log_dataset_metrics(dev_metrics, overrides_file, label="dev")
        _log_dataset_metrics(test_metrics, overrides_file, label="test")

        if is_better:
            best_score = dev_score
            best_config = param_combo
            best_metrics = dev_metrics

    with open(overrides_file, "a", encoding="utf-8") as f:
        f.write("\n=== Лучшая комбинация (Dev-based) ===\n")
        for k, v in best_config.items():
            f.write(f"{k} = {v}\n")

    logging.info("Полный перебор завершён! Лучшие параметры выбраны.")
    return best_score, best_config, best_metrics


def _log_dataset_metrics(metrics, file_path, label="dev"):
    if "by_dataset" not in metrics:
        return

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n>>> Подробные метрики по каждому датасету ({label}):\n")
        for ds in metrics["by_dataset"]:
            name = ds.get("name", "unknown")
            f.write(f"  - {name}:\n")
            for k in ["uar", "mF1", "ACC", "CCC"]:
                if k in ds:
                    f.write(f"      {k.upper():4} = {ds[k]:.4f}\n")
        f.write(f"<<< Конец подробных метрик ({label})\n")
