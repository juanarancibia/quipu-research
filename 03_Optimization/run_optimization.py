import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Literal, Optional

import dspy
import litellm
from dspy.teleprompt import BootstrapFewShot, MIPROv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_Evaluation"))
from evaluator import Evaluator
from schemas import PredictionEntry

from dspy_modules.programs import TransactionExtractor
from metrics import parse_prediction, quipu_metric
from observability import OptimizationTracker

logger = logging.getLogger("OptimizerConfig")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(ch)


def load_dspy_dataset(filepath: str) -> list[dspy.Example]:
    """Load golden dataset from a JSONL file into a list of DSPy Examples."""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            dspy_example = dspy.Example(
                message=entry['input'],
                targets=entry.get('targets', [])
            ).with_inputs('message')
            dataset.append(dspy_example)
    return dataset


def split_dataset(dataset: list[dspy.Example], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    Randomly split the dataset into Train, Validation, and Test subsets.

    Uses a fixed seed for reproducible splits across runs.
    """
    random.seed(42)
    shuffled = dataset.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    trainset = shuffled[:n_train]
    valset = shuffled[n_train:n_train + n_val]
    testset = shuffled[n_train + n_val:]

    return trainset, valset, testset


def build_lm(model_str: str, is_teacher: bool = False) -> dspy.LM:
    """Instantiate a DSPy LM from a model string."""
    # OpenAI reasoning models (o1, o3) have strict parameter requirements in DSPy
    # Pattern: 'openai/o1-preview', 'o1-mini', etc.
    is_reasoning = any(x in model_str.lower() for x in ["/o1", "/o3"]) or model_str.lower().startswith(("o1", "o3"))
    
    if is_reasoning:
        # Per DSPy 3.0+ requirements for reasoning models:
        # temperature must be 1.0 or None, max_tokens must be >= 16000 or None
        logger.info(f"Detected reasoning model {model_str}. Applying specific parameters (max_tokens=16000, temp=1.0).")
        return dspy.LM(model_str, max_tokens=16000, temperature=1.0)
    
    if is_teacher:
        # Teacher models (used mostly in MIPROv2 to generate instructions) should NOT have json_object mode,
        # otherwise the OpenAI API will hang/crash with 400 Bad Request since internal DSPy prompts don't request JSON.
        return dspy.LM(model_str, max_tokens=16000)
    else:
        # Increased max_tokens to 16000 to avoid truncation when using CoT and multiple few-shot demos
        # Enforce JSON mode to completely eliminate markdown wrapper issues for the student
        return dspy.LM(model_str, max_tokens=16000)


def build_teleprompter(
    optimizer: str,
    teacher_lm: Optional[dspy.LM],
    auto: Literal["light", "medium", "heavy"],
) -> BootstrapFewShot | MIPROv2:
    """
    Instantiate the requested DSPy teleprompter.

    Args:
        optimizer: "bootstrap" for BootstrapFewShot, "miprov2" for MIPROv2.
        teacher_lm: Optional LM used as the teacher/prompt model in MIPROv2.
        auto: MIPROv2 optimization budget ("light", "medium", or "heavy").

    Returns:
        A configured DSPy teleprompter instance.
    """
    if optimizer == "miprov2":
        return MIPROv2(
            metric=quipu_metric,
            prompt_model=teacher_lm,   # teacher: generates instructions/candidates
            task_model=None,           # student: uses the globally configured dspy LM
            auto=auto,
            verbose=True,
        )
    # Default: BootstrapFewShot
    return BootstrapFewShot(
        metric=quipu_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run DSPy optimization for the Quipu transaction extractor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LiteLLM model string for the student/task model (e.g. openrouter/google/gemma-3-12b-it:free).",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help=(
            "LiteLLM model string for the teacher/prompt model used by MIPROv2 "
            "to generate instruction candidates (e.g. openai/gpt-4o). "
            "Ignored when --optimizer is 'bootstrap'."
        ),
    )

    # Optimizer arguments
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["bootstrap", "miprov2"],
        default="bootstrap",
        help="DSPy teleprompter to use for optimization.",
    )
    parser.add_argument(
        "--auto",
        type=str,
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 optimization budget. Ignored when --optimizer is 'bootstrap'.",
    )

    # Dataset / misc arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="../golden_dataset.jsonl",
        help="Path to golden_dataset.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the dataset size for quick smoke tests.",
    )
    parser.add_argument(
        "--use-cot",
        action="store_true",
        help="Use ChainOfThought instead of Predict inside the TransactionExtractor.",
    )

    args = parser.parse_args()

    # Log chosen configuration
    logger.info(f"Optimizer : {args.optimizer}")
    logger.info(f"Student model : {args.model}")
    if args.optimizer == "miprov2":
        logger.info(f"Teacher model : {args.teacher_model or '(same as student)'}")
        logger.info(f"MIPROv2 auto  : {args.auto}")

    # Init tracker
    teleprompter_label = f"{args.optimizer}_{args.auto}" if args.optimizer == "miprov2" else args.optimizer
    tracker = OptimizationTracker(model_name=args.model, teleprompter_name=teleprompter_label)

    # Build student LM and configure DSPy globally
    student_lm = build_lm(args.model, is_teacher=False)
    dspy.settings.configure(lm=student_lm)

    # Build optional teacher LM for MIPROv2. Even if teacher is the same string,
    # we need a *separate* LM object with is_teacher=True so it doesn't use JSON mode!
    teacher_lm: Optional[dspy.LM] = None
    if args.optimizer == "miprov2":
        teacher_model_str = args.teacher_model if args.teacher_model else args.model
        teacher_lm = build_lm(teacher_model_str, is_teacher=True)

    # LiteLLM cost/usage callback
    global GLOBAL_TRACKER
    GLOBAL_TRACKER = tracker

    def local_callback(kwargs, completion_response, start_time, end_time):
        global GLOBAL_TRACKER
        if GLOBAL_TRACKER:
            try:
                usage = completion_response.get("usage", {})
                cost = litellm.completion_cost(completion_response=completion_response)
                GLOBAL_TRACKER.add_usage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost=float(cost) if cost else 0.0,
                )
            except Exception as e:
                logger.warning(f"Failed extracting cost: {e}")

    litellm.success_callback = [local_callback]

    # Load data
    logger.info("Loading dataset...")
    dataset = load_dspy_dataset(args.dataset)

    if args.limit:
        dataset = dataset[:args.limit]

    trainset, valset, testset = split_dataset(dataset)
    logger.info(f"Dataset split -> Train: {len(trainset)} | Val: {len(valset)} | Test: {len(testset)}")

    # Build program and teleprompter
    program = TransactionExtractor(use_cot=args.use_cot)
    teleprompter = build_teleprompter(
        optimizer=args.optimizer,
        teacher_lm=teacher_lm,
        auto=args.auto,
    )

    # Compile (optimize)
    logger.info("Starting compilation (Optimization) on trainset...")
    try:
        if args.optimizer == "miprov2":
            compiled_program = teleprompter.compile(
                student=program,
                trainset=trainset,
                valset=valset,
                requires_permission_to_run=False,
            )
        else:
            compiled_program = teleprompter.compile(student=program, trainset=trainset)
            
        logger.info("Compilation complete!")
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user. Saving best program so far...")
        # Accessing the best program in MIPROv2 during interruption is hard, 
        # so we'll just continue to evaluation with whatever 'program' is currently held 
        # (mostly for logging purposes).
        compiled_program = program 
    except Exception as e:
        logger.error(f"Critical error during optimization: {e}")
        return

    # Evaluate on test set
    logger.info("Evaluating program on TEST SET...")
    evaluator = Evaluator(args.dataset)
    total_score = 0.0
    test_predictions: list[PredictionEntry] = []
    
    # We will log failures specifically to understand what's going wrong
    failures_log = []

    for ex in testset:
        try:
            pred = compiled_program(message=ex.message)
            score = quipu_metric(ex, pred)
            total_score += score
            raw_response = getattr(pred, "financial_transactions_json", "")
            
            if score < 0.8:
                failures_log.append({
                    "input": ex.message,
                    "expected": ex.targets,
                    "predicted": parse_prediction(raw_response),
                    "score": score,
                    "raw": raw_response
                })
                
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            raw_response = ""

        parsed_targets = parse_prediction(raw_response)
        evaluator._golden[ex.message] = ex.targets
        test_predictions.append(
            PredictionEntry(
                input=ex.message,
                raw_response=raw_response,
                parsed_targets=parsed_targets,
            )
        )

    # Save specifically failures for the user to review
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    if failures_log:
        with open(f"logs/failures_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(failures_log, f, indent=2, ensure_ascii=False)
        logger.info(f"Logged {len(failures_log)} failure cases to logs/failures_{timestamp}.json")

    avg_score = total_score / len(testset) if testset else 0.0
    logger.info(f"Final Average Metric Score on Testset: {avg_score:.4f}")

    # Error statistics
    error_statistics: dict | None = None
    if test_predictions:
        report = evaluator.evaluate(test_predictions)
        error_statistics = dict(report["error_statistics"])

    # Save compiled program
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("compiled_programs", exist_ok=True)
    program_path = f"compiled_programs/optimized_extractor_{timestamp}.json"
    compiled_program.save(program_path)
    logger.info(f"Program saved to {program_path}")

    # Save log
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/optimization_{timestamp}.json"
    tracker.finalize_and_save(
        best_score=avg_score,
        output_path=log_path,
        best_program_path=program_path,
        error_statistics=error_statistics,
    )


if __name__ == "__main__":
    main()
